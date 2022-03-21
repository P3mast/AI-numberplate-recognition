# AI-numberplate-recognition

In assets folder you'll have multiple images to test the code in different configurations. As you will see by experimenting, the photo must be taken straight in front or behind the car. If the device that take the photo isn't quite parallel to the vehicle the AI won't be able to correclty crop the photo and then extract the text from the image.
Further more if the numberplate, isn't a strandardized one or still wrotten with characters in black with a white background after the black and white transformation of the image then the AI won't be able to extract the text.
To try to solve this problem, in that case I invert the black and white on the image but still the issue remains. 
In a futur improvment of my code I'll try to process the image in view to make the black more intensive and present to facilitate the AI process...
