## recognize webcam video 
python recognize_video.py —shape-predictor shape_predictor_68_face_landmarks.dat —embedding-model openface_nn4.small2.v1.t7 —recognizer output/recognizer.pickle —le output/le.pickle

## extract_embeddings
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --embedding-model openface_nn4.small2.v1.t7 --shape-predictor shape_predictor_68_face_landmarks.dat

## train_model
python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
