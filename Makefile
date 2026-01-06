.PHONY: all clean x86 arm64 riscv

# Define the source file
SRC = sqlite-ndvss.c

# --- x86_64 (Native Host) ---
x86:
	mkdir -p x86
	zig cc -target x86_64-linux-gnu -O3 -shared -fPIC -o ./x86/ndvss.so $(SRC) -lm
# gcc -O3 -shared -fPIC -lm -o ./x86/ndvss.so $(SRC) 

# --- AArch64 (ARM64) ---
arm64:
	mkdir -p arm64
	zig cc -target aarch64-linux-gnu -O3 -shared -fPIC -o ./arm64/ndvss.so $(SRC)

# --- RISC-V (RV64GCV) ---
riscv:
	mkdir -p riscv
	zig cc -target riscv64-linux-gnu -mcpu=generic_rv64+v -O3 -shared -fPIC -o ./riscv/ndvss.so $(SRC)

# --- Windows x64 ---
win64:
	mkdir -p win64
	zig cc -target x86_64-windows-gnu -O3 -shared -o ./win64/ndvss.dll $(SRC)

# --- Windows ARM64 ---
win-arm64:
	mkdir -p win-arm64
	zig cc -target aarch64-windows-gnu -O3 -shared -o ./win-arm64/ndvss.dll $(SRC)

# --- macOS: Apple Silicon ---
macos-arm64:
	mkdir -p macos-arm64
	zig cc -target aarch64-macos -O3 -shared -o ./macos-arm64/ndvss.dylib $(SRC)

# --- macOS: Intel (x64) ---
macos-x64:
	mkdir -p macos-x64
	zig cc -target x86_64-macos -O3 -shared -o ./macos-x64/ndvss.dylib $(SRC)

# --- Build Everything ---
all: x86 arm64 riscv win64 win-arm64 macos-arm64 macos-x64

# --- Clean Up ---
# Use -r to remove the directories and everything inside them
clean:
	rm -rf x86 arm64 riscv win64 win-arm64 macos-arm64 macos-x64 
