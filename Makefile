CC = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wno-missing-field-initializers
LDFLAGS = -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
GLFW_CFLAGS = $(shell pkg-config --cflags glfw3)
GLFW_LIBS = $(shell pkg-config --libs glfw3)
TARGET = gibson
all: $(TARGET)
$(TARGET): src/main.cpp
	$(CC) $(CXXFLAGS) $(GLFW_CFLAGS) -o $@ $< $(LDFLAGS) $(GLFW_LIBS)
run: $(TARGET)
	./$(TARGET)
clean:
	rm -f $(TARGET)
.PHONY: all run clean
