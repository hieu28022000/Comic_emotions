import cv2

# format tlwh
def face_detect_image(image, detector):
    info_face = detector.detect_faces(image)
    list_bbox = []
    for info in info_face:
        bbox = info['box']
        list_bbox.append(bbox)
    
    return list_bbox