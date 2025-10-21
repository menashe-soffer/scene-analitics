This repositary contains my solution to scene-analysis excersize.

All the pipeline can be run by just running main.py;
this will do:
(1) apply people detector and feature extractor network on the 4 video clips. this is the most time consuming step.
(2) apply weapon detector on the 4 video clips.
(3) fix the people detections using the feature extractor output
(4) generate the output video clips
(50 generate a summary (print to screan)

The results of the people detecton process are stored in the data folder; for each input clip 2 output clips are generated: *** one with the original object detector output, and one with the final (fixed) output. ***
