import cv2
import numpy as np

def find_closest_skeleton_point(skeleton, point):

    skeleton_coords = np.column_stack(np.where(skeleton == 255))
    
    if len(skeleton_coords) == 0:
        return None
    
    skeleton_points = skeleton_coords[:, [1, 0]]
    
    distances = np.linalg.norm(skeleton_points - np.array(point), axis=1)
    
    closest_idx = np.argmin(distances)
    closest_point = tuple(skeleton_points[closest_idx])
    
    return closest_point

def mark_points(image, selected_point, closest_point, selected_color=(0, 255, 0), closest_color=(255, 0, 0)):

    marked = image.copy()
    
    if selected_point is not None:
        cv2.circle(marked, selected_point, 5, selected_color, -1)
        cv2.circle(marked, selected_point, 7, selected_color, 2)
    
    if closest_point is not None:
        cv2.circle(marked, closest_point, 5, closest_color, -1)
        cv2.circle(marked, closest_point, 7, closest_color, 2)
        
        if selected_point is not None:
            cv2.line(marked, selected_point, closest_point, (255, 255, 0), 2)
    
    return marked

def main():
    image_path = "image.png"
    mask_path = "mask.png"
    overlay_color = (0, 0, 255)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # type: ignore

    skeleton = cv2.ximgproc.thinning(mask_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    result = image.copy() # type: ignore
    result[skeleton == 255] = overlay_color

    scale = 1.0
    display = cv2.resize(result, (0, 0), fx=scale, fy=scale)
    
    selected_point = None
    closest_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point, closest_point, display
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            closest_point = find_closest_skeleton_point(skeleton, selected_point)
            
            display = cv2.resize(result, (0, 0), fx=scale, fy=scale)
            display = mark_points(display, selected_point, closest_point)
            cv2.imshow("Skeleton Overlay", display)
    
    cv2.imshow("Skeleton Overlay", display)
    cv2.setMouseCallback("Skeleton Overlay", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()