VERSION=1.00
PACKAGE=libtimer-$(VERSION).tar.bz2
LIBDIR=../lib
INCDIR=../include
SRCDIR=libtimer-$(VERSION)

all: libs headers

libs: $(LIBDIR)/libtimer.a 
headers: $(INCDIR)/Timer.h

$(LIBDIR)/libtimer.a : $(SRCDIR)/Makefile
	PREFIX=`pwd`/../; \
	cd $(SRCDIR); ./configure --prefix=$$PREFIX; make; make install

$(INCDIR)/Timer.h : $(LIBDIR)/libtimer.a

$(SRCDIR)/Makefile :
	tar jxvf $(PACKAGE)

.PHONY: clean distclean

clean:
	cd $(SRCDIR) ; make clean

distclean:
	rm -rf $(SRCDIR)
	rm -f $(LIBDIR)/libtimer.a $(INCDIR)/Timer.h


