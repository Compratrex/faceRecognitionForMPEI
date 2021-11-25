import dlib
from skimage import io
from scipy.spatial import distance


class CheckPhotos():
    def __init__(self, photo1, photo2, rec_model, shape_perdictor):
        self.photo1 = photo1
        self.photo2 = photo2
        self.rec_model = rec_model
        self.shape_perdictor = shape_perdictor

    def check_two_photos(self):
        sp = dlib.shape_predictor(self.shape_perdictor)
        facerec = dlib.face_recognition_model_v1(self.rec_model)
        detector = dlib.get_frontal_face_detector()

        img = io.imread(self.photo1)

        win1 = dlib.image_window()
        win1.clear_overlay()
        win1.set_image(img)

        dets = detector(img, 1)

        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img, d)
            win1.clear_overlay()
            win1.add_overlay(d)
            win1.add_overlay(shape)

        face_descriptor1 = facerec.compute_face_descriptor(img, shape)

        img = io.imread(self.photo2)
        win2 = dlib.image_window()
        win2.clear_overlay()
        win2.set_image(img)
        dets_webcam = detector(img, 1)
        for k, d in enumerate(dets_webcam):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            shape = sp(img, d)
            win2.clear_overlay()
            win2.add_overlay(d)
            win2.add_overlay(shape)

        face_descriptor2 = facerec.compute_face_descriptor(img, shape)

        a = distance.euclidean(face_descriptor1, face_descriptor2)

        if a > 0.6:
            return False
        return True


# shape = 'shape_predictor_68_face_landmarks.dat'
# rec_mdl = 'dlib_face_recognition_resnet_model_v1.dat'
# check = CheckPhotos('elon.jpg', 'putin.jpg', rec_mdl, shape)
# print(check.check_two_photos())