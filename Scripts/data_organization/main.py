from textGrid_to_segments import *
from label_union import *
from process_labels import *

# this script takes a textGrid file with transcribed text and time-stamps, a json file with division to segments 
# and a csv file with labels from the source folders that are configured and creates one csv file per episode
# the user has to insert the subject number.

# enter subject num here
subject_num = "sub-007"

if __name__ == '__main__':
    text_to_segments(subject_num)
    label_union(subject_num)
    process_labels(subject_num)

