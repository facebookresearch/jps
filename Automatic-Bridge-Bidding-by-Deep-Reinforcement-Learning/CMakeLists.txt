cmake_minimum_required(VERSION 2.6)
PROJECT(OctaveTest)
INCLUDE_DIRECTORIES(/usr/include/
/usr/local/include/
/usr/include/octave-4.2.2/
/usr/include/octave-4.2.2/octave
)

#find_package(Octave REQUIRED)

LINK_DIRECTORIES(/usr/lib /usr/local/lib
/usr/lib/
/usr/lib/x86_64-linux-gnu/)

ADD_EXECUTABLE(OctaveTest test.cc)
TARGET_LINK_LIBRARIES(OctaveTest PUBLIC
octave octinterp
)
