import cv2
import numpy as np
from collections import deque
import torch
from torchvision import transforms
import sys
sys.path.append('..')
from resnet_unet_atten import RESNET_UNET_ATTEN

class RoadSegmentationModel:
    def __init__(self, model_path, device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = RESNET_UNET_ATTEN(in_channels=3, out_channels=1).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Define image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = image.shape[:2]
        
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)
        
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        if mask.shape != (original_height, original_width):
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        
        return mask

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

# Finds closest point on the skeleton from selected point
def find_closest_skeleton_point(skeleton, point):

    skeleton_coords = np.column_stack(np.where(skeleton == 255))
    
    if len(skeleton_coords) == 0:
        return None
    
    skeleton_points = skeleton_coords[:, [1, 0]]
    
    distances = np.linalg.norm(skeleton_points - np.array(point), axis=1)
    
    closest_idx = np.argmin(distances)
    closest_point = tuple(skeleton_points[closest_idx])
    
    return closest_point

# Find shortest path using BFS
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

# Connects broken road segments using the following method:
# Clustering -> Connect closest pixel from each cluster to another cluster
# -> uses union find to connect clusters -> iteratively repeat until no clusters left 
# unconnected
def connect_broken_roads(skeleton, max_gap_distance=30):

    # Find connected components (clusters)
    num_labels, labels = cv2.connectedComponents(skeleton)
    
    if num_labels <= 2:  # Only background and one component
        return skeleton
    
    result = skeleton.copy()
    
    # Extract coordinates for each component (skip background label 0)
    components = []
    for label in range(1, num_labels):
        coords = np.column_stack(np.where(labels == label))
        if len(coords) > 0:
            # Store as (x, y) format
            coords = coords[:, [1, 0]]
            components.append(coords)
    
    # Find closest pairs between all component pairs
    connections = []
    
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            # Find closest points between component i and j
            comp1 = components[i]
            comp2 = components[j]
            
            # Calculate all pairwise distances
            distances = np.linalg.norm(comp1[:, np.newaxis] - comp2[np.newaxis, :], axis=2)
            
            # Find minimum distance
            min_dist = np.min(distances)
            
            if min_dist <= max_gap_distance:
                # Find the indices of the closest points
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                pt1 = tuple(comp1[min_idx[0]])
                pt2 = tuple(comp2[min_idx[1]])
                
                connections.append((pt1, pt2, min_dist))
    
    # Sort connections by distance and connect them
    connections.sort(key=lambda x: x[2])
    
    # Use Union-Find to avoid creating cycles
    parent = list(range(len(components)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Build component lookup
    point_to_component = {}
    for idx, comp in enumerate(components):
        for pt in comp:
            point_to_component[tuple(pt)] = idx
    
    # Connect components
    for pt1, pt2, dist in connections:
        comp1 = point_to_component.get(pt1)
        comp2 = point_to_component.get(pt2)
        
        if comp1 is not None and comp2 is not None:
            if union(comp1, comp2):
                # Draw the connection
                cv2.line(result, pt1, pt2, 255, 1)
    
    return result


def connect_broken_roads_iterative(skeleton, max_gap_distance=30, iterations=3):
    result = skeleton.copy()
    
    for i in range(iterations):
        prev_result = result.copy()
        result = connect_broken_roads(result, max_gap_distance)
        
        # Stop if no changes were made
        if np.array_equal(result, prev_result):
            print(f"Converged after {i+1} iteration(s)")
            break
    
    return result

def main():
    image_path = "sample1.png"
    mask_path = "mask.png"  # Optional: can be None if using model prediction
    model_weights_path = "../weights/resnet_unet_atten.pth.tar"
    overlay_color = (0, 0, 255)
    use_model = True  # Set to True to use model prediction instead of loading mask
    
    # Post-processing parameters
    use_connection = False  # Enable road connection
    max_gap_distance = 100  # Maximum pixels to bridge
    connection_iterations = 3  # Number of iterations to run

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if use_model:
        print("Loading model...")
        seg_model = RoadSegmentationModel(model_weights_path)
        print("Performing inference...")
        mask = seg_model.predict(image)
        print("Inference complete!")
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # type: ignore

    skeleton = cv2.ximgproc.thinning(mask_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # Connect broken roads
    if use_connection:
        print("Connecting broken road segments...")
        skeleton = connect_broken_roads_iterative(
            skeleton, 
            max_gap_distance=max_gap_distance,
            iterations=connection_iterations
        )
        print("Connection complete!")

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
