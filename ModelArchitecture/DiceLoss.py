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
    # Ensure proper TensorFlow module is used
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
