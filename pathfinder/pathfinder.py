import cv2
import numpy as np
from collections import deque

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

def draw_path(image, path, color=(255, 0, 255), thickness=2):
    if path is None or len(path) < 2:
        return image
    
    result = image.copy()
    for i in range(len(path) - 1):
        cv2.line(result, path[i], path[i + 1], color, thickness)
    return result

def find_closest_skeleton_point(skeleton, point):

    skeleton_coords = np.column_stack(np.where(skeleton == 255))
    
    if len(skeleton_coords) == 0:
        return None
    
    skeleton_points = skeleton_coords[:, [1, 0]]
    
    distances = np.linalg.norm(skeleton_points - np.array(point), axis=1)
    
    closest_idx = np.argmin(distances)
    closest_point = tuple(skeleton_points[closest_idx])
    
    return closest_point

def find_shortest_path(skeleton, start, end):
    start = (start[1], start[0])
    end = (end[1], end[0])
    
    if skeleton[start] != 255 or skeleton[end] != 255:
        return None
    
    queue = deque([start])
    visited = {start: None}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            path = []
            node = end
            while node is not None:
                path.append((node[1], node[0]))
                node = visited[node]
            return path[::-1]
        
        for dy, dx in directions:
            ny, nx = current[0] + dy, current[1] + dx
            
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                if skeleton[ny, nx] == 255 and (ny, nx) not in visited:
                    visited[(ny, nx)] = current # type: ignore
                    queue.append((ny, nx))
    
    return None

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
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, display
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            closest_point = find_closest_skeleton_point(skeleton, selected_point)
            
            if closest_point is not None:
                points.append(closest_point)
            
            display = cv2.resize(result, (0, 0), fx=scale, fy=scale)
            
            for i, pt in enumerate(points):
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.circle(display, pt, 5, color, -1)
                cv2.circle(display, pt, 7, color, 2)
            
            if len(points) == 2:
                path = find_shortest_path(skeleton, points[0], points[1])
                if path is not None:
                    display = draw_path(display, path)
                points = []
            
            cv2.imshow("Skeleton Overlay", display)
    
    cv2.imshow("Skeleton Overlay", display)
    cv2.setMouseCallback("Skeleton Overlay", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()