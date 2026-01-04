.PHONY: all clean x86 arm64 riscv

# Define the source file
SRC = sqlite-ndvss.c

# --- x86_64 (Native Host) ---
x86:
	mkdir -p x86
	gcc -O3 -shared -fPIC -lm -o ./x86/ndvss.so $(SRC)

# --- AArch64 (ARM64) ---
arm64:
	mkdir -p arm64
	zig cc -target aarch64-linux-gnu -O3 -shared -fPIC -o ./arm64/ndvss.so $(SRC)

# --- RISC-V (RV64GCV) ---
riscv:
	mkdir -p riscv
	zig cc -target riscv64-linux-gnu -mcpu=generic_rv64+v -O3 -shared -fPIC -o ./riscv/ndvss.so $(SRC)

# --- Build Everything ---
all: x86 arm64 riscv

# --- Clean Up ---
# Use -r to remove the directories and everything inside them
clean:
	rm -rf x86 arm64 riscv