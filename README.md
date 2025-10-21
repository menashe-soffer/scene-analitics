this is a solution to the scene-analysis exercise.
It analyses the 4 video clips in the video folder.

The whole pipeline can be done by runninh main.py, which does the follows:

(1) it applies object detctor to detect people, and a feature extractor to extract feature. this is the most time consumming step.
(2) it applies an object detector trained to detect weapons (pistols and knifes).
(3) it runs a function that do some fixes and re-ID to the people detections.
(4) it cretaes output video clips.
(5) it generate text output to the screan.

#### A description of the processing steps can be found in the .pdf file in this folder.

the output videos are in the data folder. **there are 2 output clips for each input clip: one with the original detcetions, and one with the final (fixed) detections**.

the text output is read as follows:

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
