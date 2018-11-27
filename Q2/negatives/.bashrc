#
# Initialise enivironment variables such as path, MANPATH to site standards
#
# Please do not remove the following line you can make adjustments to the 
# standard paths below if you need to.
#

source /usr/local/lib/bash.env

#
# You can modify the standard paths here eg.
#
# PATH=~/bin:$PATH
# MANPATH=$MANPATH:$HOME/man
#

#
# Standard interactive start up. Display motd, set prompt, etc.
#
# Again please do not remove these lines or you will miss
# important messages.
#
if [ -z "$PS1" ]
then
        return
fi

source /usr/local/lib/bash.int

#
# make changes to your interactive environment here:
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/opencv2/:/usr/include/opencv
export OPENCV_CFLAGS=-I/usr/include/opencv2/
export O_LIBS="-L/usr/lib64/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann"
