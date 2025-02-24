import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def plot_gt_pred(pred, gt):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap='gray')
    plt.title('Predicted')
    plt.axis('off')  

    plt.subplot(1, 2, 2)
    plt.imshow(gt, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off') 
    plt.show()


def plot_contours(predicted_contours_image, contours_pred, ground_truth_contours_image, contours_gt):

    cv2.drawContours(predicted_contours_image, contours_pred, -1, (0, 255, 0), 2)  # Green for predicted
    cv2.drawContours(ground_truth_contours_image, contours_gt, -1, (255, 0, 0), 2)  # Blue for ground truth

    # Convert the images from BGR to RGB for Matplotlib plotting
    predicted_contours_image_rgb = cv2.cvtColor(predicted_contours_image, cv2.COLOR_BGR2RGB)
    ground_truth_contours_image_rgb = cv2.cvtColor(ground_truth_contours_image, cv2.COLOR_BGR2RGB)

    # Plot the images with contours
    plt.figure(figsize=(15, 7))

    # Predicted label visualization 
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_contours_image_rgb, cmap='gray')
    plt.title('Predicted Contours')
    plt.axis('off')  # Hide axes

    # Ground truth label visualization
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth_contours_image_rgb, cmap='gray')
    plt.title('Ground Truth Contours')
    plt.axis('off')  # Hide axes
    plt.show()


def calculate_dice(predicted, ground_truth):
    # Flatten the arrays to 1D
    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()

    # Calculate the intersection and sum
    intersection = np.sum(predicted * ground_truth)
    sum = np.sum(predicted) + np.sum(ground_truth)

    # Calculate the Dice coefficient
    dice = 2 * intersection / sum if sum > 0 else 0  # Avoid division by zero
    return dice


def calculate_NSD(contours_pred, contours_gt, image_shape, dilation_radius=1):
    # Create binary masks for the predicted and ground truth contours
    pred_mask = np.zeros(image_shape, dtype=np.uint8)
    gt_mask = np.zeros(image_shape, dtype=np.uint8)

    # Draw contours on the masks
    cv2.drawContours(pred_mask, contours_pred, -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(gt_mask, contours_gt, -1, 255, thickness=cv2.FILLED)

    # Dilate the masks to create a tolerance region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1))
    dilated_gt_mask = cv2.dilate(gt_mask, kernel)
    dilated_pred_mask = cv2.dilate(pred_mask, kernel)

    # Count points within the dilated masks
    pred_in_dilated_gt = np.sum((pred_mask > 0) & (dilated_gt_mask > 0))
    gt_in_dilated_pred = np.sum((gt_mask > 0) & (dilated_pred_mask > 0))

    # Total number of contour points
    total_pred_points = np.sum(pred_mask > 0)
    total_gt_points = np.sum(gt_mask > 0)

    # Compute NSD
    nsd_pred = pred_in_dilated_gt / total_pred_points if total_pred_points > 0 else 0
    nsd_gt = gt_in_dilated_pred / total_gt_points if total_gt_points > 0 else 0

    # Final NSD as an average of both directions
    nsd = (nsd_pred + nsd_gt) / 2

    return nsd


def calculate_NSD_MASD(predicted, ground_truth):
    # Scale to 0-255 range
    predicted = (predicted / np.max(predicted) * 255).astype(np.uint8)
    ground_truth = (ground_truth / np.max(ground_truth) * 255).astype(np.uint8)

    # Create color versions of the binary images for visualization
    predicted_contours_image = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)
    ground_truth_contours_image = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)

    # Extract contours for the predicted and ground truth slices
    contours_pred, _ = cv2.findContours(predicted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(ground_truth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Plot the contours for visualisation
    plot_contours(predicted_contours_image, contours_pred, ground_truth_contours_image, contours_gt)

    total_distance = 0.0
    total_points = 0

    # Iterate over all predicted contours
    for pred_contour in contours_pred:
        if len(contours_pred) != 0:
            pred_points = np.vstack(pred_contour)  # Flatten current predicted contour
            # Find minimum distance to all ground truth points
            distances = [min(distance.cdist([p], np.vstack(gt_contour)).min() for gt_contour in contours_gt) for p in pred_points]
            total_distance += sum(distances)
            total_points += len(pred_points)

    # Iterate over all ground truth contours
    for gt_contour in contours_gt:
        if len(contours_gt) != 0:
            gt_points = np.vstack(gt_contour)  # Flatten current ground truth contour
            # Find minimum distance to all predicted points
            distances = [min(distance.cdist([g], np.vstack(pred_contour)).min() for pred_contour in contours_pred) for g in gt_points]
            total_distance += sum(distances)
            total_points += len(gt_points)

    # Compute MASD and NASD
    masd = total_distance / total_points if total_points > 0 else 0  # Avoid division by zero
    nsd = calculate_NSD(contours_pred, contours_gt, predicted.shape)  
    return masd, nsd


if __name__ == "__main__":
    # Load the .mha file using SimpleITK
    image_path = '../BONBID2023_Train/3LABEL/MGHNICU_062-VISIT_01_lesion.mha'  # Replace with your file path
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)  # Convert to a numpy array (shape: [depth, height, width])

    predicted = image_array[11]
    ground_truth = image_array[10]  

    # Plotting the predicted and ground truth images
    plot_gt_pred(predicted, ground_truth)

    # Calculating performance metrics
    masd, nsd = calculate_NSD_MASD(predicted, ground_truth)
    dice = calculate_dice(predicted, ground_truth)
    print(f'Mean Average Surface Distance (MASD): {masd}')
    print(f'Normalized Surface Distance (NSD): {nsd}')
    print(f'Dice Coefficient: {dice}')



    
