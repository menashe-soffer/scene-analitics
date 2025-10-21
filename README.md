This repositary contains my solution to scene-analysis excersize.

All the pipeline can be run by just running main.py;
this will do:
(1) apply people detector and feature extractor network on the 4 video clips. this is the most time consuming step.
(2) apply weapon detector on the 4 video clips.
(3) fix the people detections using the feature extractor output
(4) generate the output video clips
(50 generate a summary (print to screan)

The results of the people detecton process are stored in the data folder; for each input clip 2 output clips are generated:  **one with the original object detector output, and one with the final (fixed) output**.



the text output is:


```
summary of person detections

ID		 clip-1  clip-2   clip-3   clip-4

1  :	 604       64      530              
2  :	 612       73      257              
3  :	  53                                
4  :	  55                                
6  :	         1071      601              
7  :	          845                       
8  :	          921                       
9  :	           12                       
14  :	                   577              
17  :	                   356              
18  :	                    22              
19  :	                    13              
21  :	                            807     
22  :	                            612     
23  :	                             31     


summary of crime scene detection

C:\Users\menas\OneDrive\Desktop\hw-felix\videos\1.mp4: some weapon detect in 0 frames,  slim probability for acrime
C:\Users\menas\OneDrive\Desktop\hw-felix\videos\2.mp4: some weapon detect in 58 frames,  low probability for acrime
C:\Users\menas\OneDrive\Desktop\hw-felix\videos\3.mp4: some weapon detect in 0 frames,  slim probability for acrime
C:\Users\menas\OneDrive\Desktop\hw-felix\videos\4.mp4: some weapon detect in 252 frames,  high probability for acrime

```
