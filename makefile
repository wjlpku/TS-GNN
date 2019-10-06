TF_CFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS = $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
PYPATH = -L${HOME}/Lib/anaconda3/lib -I${HOME}/Lib/anaconda3/include/python3.6m -lpython3.6m
CFLAGS = -Wall -Wextra -DLOCAL -Wshadow -Wno-unused-result -Wpointer-arith -Wcast-qual -Wunreachable-code

all:
	g++ -std=c++11 -shared draw.cc -o draw.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
	g++ -std=c++11 -shared repeat.cc -o repeat.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
	g++ -std=c++11 -shared reader.cc -o reader.so -fPIC -fopenmp $(PYPATH) $(CFLAGS) -O2
	g++ -std=c++11 -shared alias.cc -o alias.so -fPIC -fopenmp $(PYPATH) $(CFLAGS) -O2

pack:
	tar -zcvf ts_gnn.tar.gz *.cc *.py makefile .gitignore

clean:
	rm -f *.so
