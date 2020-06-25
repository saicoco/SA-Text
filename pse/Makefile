#CXXFLAGS = -I include  -std=c++11 -O3
#CXXFLAGS = -I include -std=c++14  -O3
CXXFLAGS = -I include -std=c++14 -O3 -Wl,-undefined,dynamic_lookup
DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp
#OPENCV = `pkg-config --cflags --libs opencv`
OPENCV = `pkg-config --cflags --libs /usr/local/Cellar/opencv/4.1.2/lib/pkgconfig/opencv4.pc`
LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
