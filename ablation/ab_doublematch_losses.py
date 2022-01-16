# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from tqdm import trange

from cta.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment

FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class DoubleMatch_LOSSES(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, gen_labeled, gen_unlabeled)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def model(self, batch, lr, wd, wu, ws, confidence, uratio, cosinedecay,
              uloss, softmaxtmp, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels

        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * cosinedecay * np.pi / 2) # fixmatch uses 7/8 for cosinedecay
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: tuple(self.classifier(x, **kw, **kwargs).values())
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2*uratio + 1)
        
        logits, embeds = utils.para_cat(lambda x: classifier(x, training=True), x)

        logits = utils.de_interleave(logits, 2*uratio+1)
        embeds = utils.de_interleave(embeds, 2*uratio+1)

        # project embeds of strong augmentations
        projected_embeds = utils.para_cat(lambda x: projection_head(x, **kwargs),
                                          embeds[-batch*uratio:])
        
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]

        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:],2)
        embeds_weak = embeds[batch:-batch*uratio]

        del logits, embeds, skip_ops

             
        # Labeled cross-entropy
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
                                                                  logits=logits_strong)
        mask = tf.reduce_max(pseudo_labels, axis=1) >= confidence
        pseudo_mask = tf.to_float(mask)
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)
    

        if uloss == "mse":
            loss_us = tf.reduce_mean((tf.stop_gradient(embeds_weak) - projected_embeds)**2, axis=-1)
            loss_us = tf.reduce_mean(loss_us)

        elif uloss == "softmax":
            teacher_labels = tf.nn.softmax(tf.stop_gradient(embeds_weak)/softmaxtmp)
            loss_us = tf.nn.softmax_cross_entropy_with_logits(labels=teacher_labels,
                                                              logits=projected_embeds)
            loss_us = tf.reduce_mean(loss_us)

        elif uloss == "cosine":
            normalized_proj_embeds = tf.linalg.l2_normalize(projected_embeds, axis=-1)
            normalized_embeds = tf.linalg.l2_normalize(tf.stop_gradient(embeds_weak), axis=-1)
            cosine_similarity = tf.reduce_sum(tf.multiply(normalized_proj_embeds, normalized_embeds), axis=1)
            loss_us = tf.reduce_mean(-cosine_similarity+1)

        else:
            raise ValueError("Loss {} is not implemented".format(uloss))

        
        tf.summary.scalar('losses/us', loss_us)

        
        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)


        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)     
        post_ops.append(ema_op)


        # helper function
        def _compute_gradients(tensor, var_list, colocate_gradients_with_ops):
            grads = tf.gradients(tensor, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)
            return [grad if grad is not None else tf.zeros_like(var)
                    for var, grad in zip(var_list, grads)]
        
        # evaluate loss gradient magnitudes
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        losses = {"xe": loss_xe,
                  "xeu": loss_xeu,
                  "wd": loss_wd,
                  "us": loss_us}

        loss_grads = [_compute_gradients(loss, trainable_vars, colocate_gradients_with_ops=True)
                      for loss in losses.values()]

        for key, loss_grad in zip(losses.keys(), loss_grads):
            magnitude = tf.math.sqrt(sum([tf.reduce_sum(tf.math.square(var)) for var in loss_grad]))
            tf.summary.scalar('loss_gradient_magnitudes/{}'.format(key), magnitude)

            
        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wd * loss_wd + ws * loss_us, colocate_gradients_with_ops=True)
        
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)[0]),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)[0]))



def projection_head(embeds_strong, **kwargs):
    
    with tf.variable_scope('classify', reuse=tf.AUTO_REUSE):
        shape = embeds_strong.get_shape().as_list()
        projection_embeds = tf.layers.dense(embeds_strong,
                                            shape[-1],
                                            kernel_initializer=tf.glorot_normal_initializer(),
                                            name="projection_head")
    return projection_embeds


    
def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = DoubleMatch_LOSSES(
        os.path.join(FLAGS.train_dir, dataset.name, DoubleMatch_LOSSES.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        ws=FLAGS.ws,
        cosinedecay=FLAGS.cosinedecay,
        uloss=FLAGS.uloss,
        softmaxtmp=FLAGS.softmaxtmp)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    flags.DEFINE_float('cosinedecay', 7/8, 'Factor in cosine decay for learning rate')
    flags.DEFINE_float('ws', 1, "Self supervised loss weight")
    flags.DEFINE_enum('uloss', 'cosine', ['cosine', 'mse', 'softmax'], 'Unsupervised loss function')
    flags.DEFINE_float('softmaxtmp', 1.0, 'Temperature for unsupervised softmax loss')
    app.run(main)
