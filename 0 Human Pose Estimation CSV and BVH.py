%reset -f
import cv2
import mediapipe as mp
import csv
import numpy as np
# from some_library import audio_classifier

# mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# video capture and writers.
cap = cv2.VideoCapture('test.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_annotated = cv2.VideoWriter('HPS_output_annotated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
output_skeleton = cv2.VideoWriter('HPS_output_skeleton.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

# Joint names in order according to the MediaPipe Pose model.
joint_names = [
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer',
    'Right Eye Inner', 'Right Eye', 'Right Eye Outer', 'Left Ear',
    'Right Ear', 'Mouth Left', 'Mouth Right', 'Left Shoulder',
    'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist',
    'Right Wrist', 'Left Pinky', 'Right Pinky', 'Left Index',
    'Right Index', 'Left Thumb', 'Right Thumb', 'Left Hip',
    'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle',
    'Right Ankle', 'Left Heel', 'Right Heel', 'Left Foot Index',
    'Right Foot Index'
]

# Define the skeleton hierarchy for calculating rotations
# Each entry is (child_joint, parent_joint)
hierarchy = {
    'Left Eye Inner': 'Nose',
    'Left Eye': 'Left Eye Inner',
    'Left Eye Outer': 'Left Eye',
    'Right Eye Inner': 'Nose',
    'Right Eye': 'Right Eye Inner',
    'Right Eye Outer': 'Right Eye',
    'Left Ear': 'Left Eye Outer',
    'Right Ear': 'Right Eye Outer',
    'Mouth Left': 'Nose',
    'Mouth Right': 'Nose',
    'Left Shoulder': 'Spine',
    'Right Shoulder': 'Spine',
    'Left Elbow': 'Left Shoulder',
    'Right Elbow': 'Right Shoulder',
    'Left Wrist': 'Left Elbow',
    'Right Wrist': 'Right Elbow',
    'Left Pinky': 'Left Wrist',
    'Right Pinky': 'Right Wrist',
    'Left Index': 'Left Wrist',
    'Right Index': 'Right Wrist',
    'Left Thumb': 'Left Wrist',
    'Right Thumb': 'Right Wrist',
    'Left Hip': 'Spine',
    'Right Hip': 'Spine',
    'Left Knee': 'Left Hip',
    'Right Knee': 'Right Hip',
    'Left Ankle': 'Left Knee',
    'Right Ankle': 'Right Knee',
    'Left Heel': 'Left Ankle',
    'Right Heel': 'Right Ankle',
    'Left Foot Index': 'Left Heel',
    'Right Foot Index': 'Right Heel'
}

# Define basic offsets for each joint relative to its parent in the initial bind pose
initial_offsets = {
    'Nose': np.array([0, 0, 0]),
    'Spine': np.array([0, 0.1, 0]),
    'Left Shoulder': np.array([-0.1, 0.1, 0]),
    'Right Shoulder': np.array([0.1, 0.1, 0]),
    'Left Elbow': np.array([-0.2, 0, 0]),
    'Right Elbow': np.array([0.2, 0, 0]),
    'Left Wrist': np.array([-0.2, 0, 0]),
    'Right Wrist': np.array([0.2, 0, 0]),
    'Left Hip': np.array([-0.1, -0.1, 0]),
    'Right Hip': np.array([0.1, -0.1, 0]),
    'Left Knee': np.array([-0.2, -0.3, 0]),
    'Right Knee': np.array([0.2, -0.3, 0]),
    'Left Ankle': np.array([-0.1, -0.4, 0]),
    'Right Ankle': np.array([0.1, -0.4, 0]),
    # Add other joints as necessary
}

# write the BVH header
def write_bvh_header(bvh_file, hierarchy, initial_offsets):
    def write_joint(bvh_file, joint_name, depth=1):
        offset = initial_offsets.get(joint_name, np.array([0, 0, 0]))
        indent = "  " * depth
        bvh_file.write(f"{indent}JOINT {joint_name}\n")
        bvh_file.write(f"{indent}{{\n")
        bvh_file.write(f"{indent}  OFFSET {offset[0]} {offset[1]} {offset[2]}\n")
        bvh_file.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
        for child_joint, parent_joint in hierarchy.items():
            if parent_joint == joint_name:
                write_joint(bvh_file, child_joint, depth + 1)
        bvh_file.write(f"{indent}}}\n")

    bvh_file.write("HIERARCHY\n")
    bvh_file.write("ROOT Hips\n")
    bvh_file.write("{\n")
    bvh_file.write("  OFFSET 0.0 0.0 0.0\n")
    bvh_file.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")

    # Write the joint hierarchy starting from the root's children
    for joint, parent in hierarchy.items():
        if parent == 'Spine':  # Start with joints that are direct children of the root
            write_joint(bvh_file, joint, 2)
    bvh_file.write("}\n")

# write the BVH motion data
def write_bvh_motion(bvh_file, frames, hierarchy):
    bvh_file.write("MOTION\n")
    bvh_file.write(f"Frames: {len(frames)}\n")
    bvh_file.write("Frame Time: 0.0333333\n")  # 30 FPS

    for frame_data in frames:
        frame_string = []
        for joint, data in frame_data.items():
            if joint == 'Spine':  # Root joint includes position
                frame_string.append(f"{data['position'][0]} {data['position'][1]} {data['position'][2]}")
            frame_string.extend([f"{data['rotation'][2]} {data['rotation'][0]} {data['rotation'][1]}"])  # Zrotation Xrotation Yrotation
        bvh_file.write(" ".join(frame_string) + "\n")

# calculate rotation and offset
def calculate_rotation_and_offset(parent_pos, child_pos):
    # Calculate the vector from parent to child
    direction_vector = child_pos - parent_pos
    direction_vector /= np.linalg.norm(direction_vector)  # Normalize the vector

    # Calculate rotation angles (for simplicity, rotations around fixed axes)
    pitch = np.arcsin(-direction_vector[1])
    yaw = np.arctan2(direction_vector[0], direction_vector[2])
    roll = np.arctan2(direction_vector[1], direction_vector[0])  # Simple roll calculation

    # Calculate the offset
    offset = child_pos - parent_pos

    return offset, np.degrees([pitch, yaw, roll])

frame_count = 0
frames_data = []

# Use 'with' to ensure files are properly closed
with open('HPS_joint_data_with_rotation_and_offset.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Joint', 'X', 'Y', 'Z', 'OffsetX', 'OffsetY', 'OffsetZ', 'Pitch', 'Yaw', 'Roll'])

    with open('HPS.bvh', 'w') as bvh_file:

        # Process each video frame.
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert the frame to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Copy the original frame for the annotated video.
            annotated_image = frame.copy()

            # Create a blank image for the skeleton video.
            skeleton_image = 255 * np.ones_like(frame)

            if results.pose_landmarks:
                # Store positions of the joints in a dictionary
                joint_positions = {}
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    joint_positions[joint_names[i]] = np.array([x, y, z])

                # Draw pose landmarks on the annotated image.
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                # Draw the same landmarks on the skeleton image.
                mp_drawing.draw_landmarks(
                    skeleton_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                )

                # Loop through each joint to calculate and save rotation and offset data
                frame_data = {}
                for joint, parent_joint in hierarchy.items():
                    if parent_joint in joint_positions:
                        parent_pos = joint_positions[parent_joint]
                        child_pos = joint_positions[joint]
                        offset, (pitch, yaw, roll) = calculate_rotation_and_offset(parent_pos, child_pos)
                    else:
                        offset = np.array([0, 0, 0])  # Root joints have no offset relative to a non-existent parent
                        pitch, yaw, roll = 0, 0, 0  # Root joints or missing parents have no rotation

                    # Get the coordinates of the current joint
                    x, y, z = joint_positions[joint]

                    # Annotate the joint name on the annotated image.
                    pos_x = int(x * frame_width)
                    pos_y = int(y * frame_height)
                    cv2.putText(annotated_image, joint, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(skeleton_image, joint, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                    # Write the joint data to the CSV file.
                    csv_writer.writerow([frame_count, joint, x, y, z, offset[0], offset[1], offset[2], pitch, yaw, roll])

                    # Store joint rotation data for the BVH motion section
                    frame_data[joint] = {
                        'position': np.array([x, y, z]),
                        'rotation': np.array([pitch, yaw, roll])
                    }

                frames_data.append(frame_data)

            # Write the frames into the output videos.
            output_annotated.write(annotated_image)
            output_skeleton.write(skeleton_image)

            # Display the annotated video.
            cv2.imshow('Annotated Pose Estimation', annotated_image)

            # Display the skeleton video.
            cv2.imshow('Skeleton Pose Estimation', skeleton_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the BVH header and motion sections
        write_bvh_header(bvh_file, hierarchy, initial_offsets)
        write_bvh_motion(bvh_file, frames_data, hierarchy)

# Release resources
cap.release()
output_annotated.release()
output_skeleton.release()
cv2.destroyAllWindows()
