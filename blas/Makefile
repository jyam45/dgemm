BLAS_VERSION=3.8.0
BLAS_PACKAGE=blas-$(BLAS_VERSION).tar.xz
CBLAS_PACKAGE=cblas.tar.xz
LIBDIR=../lib
INCDIR=../include
BLASDIR=BLAS-$(BLAS_VERSION)

all: libs headers

libs: $(LIBDIR)/libblas.a $(LIBDIR)/libcblas.a 
headers: $(INCDIR)/cblas.h $(INCDIR)/cblas_f77.h 

$(LIBDIR)/libblas.a : $(BLASDIR)/Makefile config/blas/make.inc
	cp config/blas/make.inc $(BLASDIR)/ ; \
	cd $(BLASDIR)/ ; \
	make 

$(LIBDIR)/libcblas.a : CBLAS/Makefile config/cblas/Makefile.in
	cp config/cblas/Makefile.in CBLAS/ ; \
	cd CBLAS/ ; \
	make alllib 

$(INCDIR)/cblas.h : CBLAS/Makefile
	cp CBLAS/include/cblas.h $(INCDIR)/


$(INCDIR)/cblas_f77.h : CBLAS/Makefile
	cp CBLAS/include/cblas_f77.h $(INCDIR)/

$(BLASDIR)/Makefile :
	tar Jxvf $(BLAS_PACKAGE)

CBLAS/Makefile:
	tar Jxvf $(CBLAS_PACKAGE)

.PHONY: clean distclean


clean:
	rm -rf $(BLASDIR)/*.o
	cd CBLAS/; make clean

distclean:
	rm -rf $(BLASDIR) CBLAS
	rm -f $(LIBDIR)/libblas.a $(LIBDIR)/libcblas.a $(INCDIR)/cblas.h $(INCDIR)/cblas_f77.h


