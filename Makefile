CXXFLAGS = $(shell pkg-config --cflags opencv)
LDLIBS = $(shell pkg-config --libs opencv)

VLROOT = /home/wangs/vlfeat-0.9.20

CC = g++

objects = main.cpp segment.cpp slic.cpp slic.h dsift.cpp dsift.h\
          dictionary.cpp dictionary.h utils.cpp utils.h llc.cpp llc.h \
          predict.cpp predict.h liblinear/linear.cpp liblinear/tron.cpp \
          liblinear/blas/daxpy.c liblinear/blas/ddot.c \
          liblinear/blas/dnrm2.c liblinear/blas/dscal.c

classify : $(objects)
	$(CC) -o classify $(objects) $(CXXFLAGS) $(LDLIBS) -I$(VLROOT) -L$(VLROOT)/bin/glnxa64/ -lvl
clean:
	/bin/rm -f classify *.o

clean-all: clean
	/bin/rm -f *~ 

