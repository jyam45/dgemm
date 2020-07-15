CC=gcc
LIBDIR=../lib
INCDIR=../include

all   : libcpuid.a

libcpuid.a: cpuid.o
	ar cru $@ $^

install: $(LIBDIR) $(INCDIR) libcpuid.a cpuid.h cache.h
	cp libcpuid.a $(LIBDIR)/
	cp cpuid.h $(INCDIR)/
	cp cache.h $(INCDIR)/

$(LIBDIR):
	mkdir -p $@

$(INCDIR):
	mkdir -p $@

cpuid.o : cpuid.c cpuid.h cache.h

.PHONY: clean distclean

clean:
	rm -f *.o

distclean: clean
	rm -r $(LIBDIR)/libcpuid.a $(INCDIR)/cpuid.h $(INCDIR)/cache.h

.SUFFIXES: .o .c
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
