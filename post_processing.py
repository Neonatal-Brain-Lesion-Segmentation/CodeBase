def doublet_thresholding(predicted, top_score_threshold, min_contour_area):
    """
    Applies doublet thresholding to filter regions in the predicted mask.
    
    Args:
        predicted (numpy.ndarray): The predicted mask, a 4D array (batch, channels, height, width).
        top_score_threshold (float): Threshold to select regions.
        min_contour_area (int): Minimum area (in pixels) for a region to be retained.
    
    Returns:
        numpy.ndarray: The filtered binary mask.
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()

    # Step 1: Apply the top threshold to create a binary classification mask
    classification_mask = predicted > top_score_threshold

    # Step 2: Filter out regions below the minimum contour area
    mask = predicted.copy()
    mask[classification_mask.sum(axis=(1, 2, 3)) < min_contour_area, :, :, :] = 0

    # Return the binary mask
    return mask > top_score_threshold

def triplet_thresholding(predicted, top_score_threshold, min_contour_area, bottom_score_threshold):
    """
    Applies triplet thresholding to refine and filter regions in the predicted mask.
    
    Args:
        predicted (numpy.ndarray): The predicted mask, a 4D array (batch, channels, height, width).
        top_score_threshold (float): Initial threshold to filter regions.
        min_contour_area (int): Minimum area (in pixels) for a region to be retained.
        bottom_score_threshold (float): Secondary threshold to refine the mask.
    
    Returns:
        numpy.ndarray: The refined binary mask.
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()

    # Step 1: Apply the top threshold to create a binary classification mask
    classification_mask = predicted > top_score_threshold

    # Step 2: Remove regions below the minimum contour area
    mask = predicted.copy()
    mask[classification_mask.sum(axis=(1, 2, 3)) < min_contour_area, :, :, :] = 0

    # Step 3: Apply the bottom threshold to refine the remaining regions
    refined_mask = mask > bottom_score_threshold

    return refined_mask