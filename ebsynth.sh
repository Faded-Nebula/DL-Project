export PATH=ebsynth/bin:$PATH

ebsynth -style ./raw_frames/frame_1.png \
-guide ./raw_frames/frame_2.png ./refined_frames/frame_2.png -weight 2.0 \
-guide raw_depth.png refined_depth.png -weight 1.5 \
-output output2.png
