'''Real Time Beat Tracker implementd on Embedded System.
Optimized to run on RPi. Should also run smoothly on systems with better
system configuration with necessary dependencies. 
Tested on both Ubuntu and Windows. 
'''

from BeatTracker import BeatTracker
#To Send the beat output to arduino for visual output instead of printing on terminal
#import serial

# sample.wav and sample.txt are default file and provided with the package for illustration
in_file = 'sample.wav'
out_file = 'sample.txt'

#Create a BeatTracker process to get real time output in text file/terminal
proc = BeatTracker()

################################################
#Only required if output from Arduino is desired
#To setup serial communication with Arduino
#portName = '/dev/ttyACM0' #find portName with ls/dev/tty* from terminal
# mySerial = serial.Serial(portName, 9600)
# proc = BeatTracker(output = 'serial', serialName = mySerial)

#Evaluation Method 1
##############################################################
#load wav file, take data frame by frame and Return beat time stamps. 
#Also save the output in a text file in the . 
# b = proc.Beats(InFile = in_file, OutFile = out_file)

#Evaluation Method 2
##############################################################
#Correct Audio Input/Output Device must be made selected in OS
# Take Real Time Audio Input form microphone and print the beat times on terminal instantaneously
# proc.MicrophoneStream()

#Evaluation Method 3
##############################################################
#Correct Audio Input/Output Device must be made selected in OS
# # Open a audio stream hence gets data in Real time and show the beat times instantaneously on terminal.
# # Also save the data in a text file
proc.wavStream(InFile= in_file, OutFile = out_file)