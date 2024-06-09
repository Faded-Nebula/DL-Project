export PATH=ebsynth/bin:$PATH

ebsynth -style bin/raw_frames/frame_16.png \
-guide bin/raw_frames/frame_4.png bin/refined_key_frames/frame_4.png -weight 2.5 \
-guide bin/gray_raw_key_frames/frame_4.png bin/gray_refined_key_frames/frame_4.png -weight 1.5 \
-output output4.png \
-patchsize 3 \
-searchvoteiters 20 \
#-extrapass3x3 