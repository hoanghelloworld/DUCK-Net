import tensorflow as tf


def dice_metric_loss(ground_truth, predictions, smooth=1e-6):
    """
    Dice loss for image segmentation task.
    The dice loss is calculated as 1 - dice coefficient.
    
    Args:
        ground_truth: Ground truth binary masks
        predictions: Predicted probability masks
        smooth: Small constant to avoid division by zero
        
    Returns:
        Dice loss value
    """
    # Try to use tf.keras.backend if available, otherwise fall back to tf operations
    try:
        # Use Keras backend for operations
        import tensorflow.keras.backend as K
        
        # Cast to float32
        ground_truth = K.cast(ground_truth, 'float32')
        predictions = K.cast(predictions, 'float32')
        
        # Flatten the tensors
        ground_truth = K.flatten(ground_truth)
        predictions = K.flatten(predictions)
        
        # Calculate intersection and union
        intersection = K.sum(ground_truth * predictions)
        union = K.sum(ground_truth) + K.sum(predictions)
        
        # Calculate dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
    except (ImportError, AttributeError):
        # Fall back to native TensorFlow operations
        # Cast to float32
        ground_truth = tf.cast(ground_truth, tf.float32)
        predictions = tf.cast(predictions, tf.float32)
        
        # Flatten the tensors
        ground_truth = tf.reshape(ground_truth, [-1])
        predictions = tf.reshape(predictions, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(ground_truth * predictions)
        union = tf.reduce_sum(ground_truth) + tf.reduce_sum(predictions)
        
        # Calculate dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return dice loss
    return 1.0 - dice
