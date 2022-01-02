import tensorflow as tf

class QARetrieverLoss(tf.keras.losses.Loss):
    def __init__(self, name = 'QARetrieverLoss', reduction = None, ** kwargs):
        super().__init__(name = name, reduction = 'none', ** kwargs)
    
    @property
    def metric_names(self):
        return ['loss', 'start_loss', 'end_loss']
    
    def call(self, y_true, y_pred):
        true_start, true_end = y_true
        pred_start, pred_end = y_pred
        
        start_loss  = tf.keras.losses.sparse_categorical_crossentropy(true_start, pred_start)
        end_loss    = tf.keras.losses.sparse_categorical_crossentropy(true_end, pred_end)
        
        return tf.stack([start_loss + end_loss, start_loss, end_loss], 0)
