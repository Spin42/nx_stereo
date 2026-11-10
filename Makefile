# Builds stereo_match, the native backend's Port, into the app's priv/: for
# the workstation, or for a Nerves target with the toolchain and sysroot
# Nerves exports while it compiles the dependency.

PRIV_DIR := $(MIX_APP_PATH)/priv
NATIVE_DIR := $(CURDIR)/native
CARGO ?= cargo

ifeq ($(filter-out host,$(MIX_TARGET)),)
RUST_OUT := $(NATIVE_DIR)/target/release

cargo:
	cd $(NATIVE_DIR) && $(CARGO) build --release --locked -p stereo_match
else
SYSROOT := $(NERVES_SDK_SYSROOT)
RUST_TARGET := $(TARGET_ARCH)-unknown-$(TARGET_OS)-$(TARGET_ABI)
RUST_TARGET_ENV := $(subst -,_,$(RUST_TARGET))
RUST_TARGET_ENV_UPPER := $(shell echo $(RUST_TARGET_ENV) | tr a-z A-Z)
RUST_OUT := $(NATIVE_DIR)/target/$(RUST_TARGET)/release
# Nerves names the CPU with underscores (cortex_a76); rustc with dashes.
TARGET_CPU_FLAG := $(if $(TARGET_CPU),-C target-cpu=$(subst _,-,$(TARGET_CPU)))

# The Nerves CC and flags would also apply to build scripts on the host, so
# they are passed only under target-scoped names.
CARGO_ENV := env -u CC -u CXX -u AR -u CFLAGS -u CXXFLAGS -u LDFLAGS \
	CC_$(RUST_TARGET_ENV)="$(CC)" \
	AR_$(RUST_TARGET_ENV)="$(AR)" \
	CARGO_TARGET_$(RUST_TARGET_ENV_UPPER)_LINKER="$(CC)" \
	CARGO_TARGET_$(RUST_TARGET_ENV_UPPER)_RUSTFLAGS="-C link-arg=--sysroot=$(SYSROOT) $(TARGET_CPU_FLAG)"

cargo:
	@test -n "$(SYSROOT)" || { echo "NERVES_SDK_SYSROOT is unset: build a target through its firmware project" >&2; exit 1; }
	cd $(NATIVE_DIR) && $(CARGO_ENV) $(CARGO) build --release --locked --target $(RUST_TARGET) -p stereo_match
endif

# Mix links a path dependency's build priv to its source priv, which the
# host and every target would then share: the build gets a priv of its own.
all: cargo
	if [ -L "$(PRIV_DIR)" ]; then rm "$(PRIV_DIR)" && mkdir -p "$(PRIV_DIR)" && cp -R $(CURDIR)/priv/. "$(PRIV_DIR)/"; fi
	mkdir -p $(PRIV_DIR)
	install -m 755 $(RUST_OUT)/stereo_match $(PRIV_DIR)/stereo_match

clean:
	rm -f $(PRIV_DIR)/stereo_match

.PHONY: all cargo clean
