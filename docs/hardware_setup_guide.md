# Raspberry Pi 5 & Hailo-8L AI HAT+ Physical Deployment Guide

This guide provides an end-to-end operational manual for assembling, configuring, verifying, and maintaining the **Raspberry Pi 5 with Hailo-8L AI HAT+** hardware accelerator platform.

---

## Table of Contents

1. [Hardware Architecture Overview](#1-hardware-architecture-overview)
2. [Physical Assembly Procedure](#2-physical-assembly-procedure)
3. [Operating System & Kernel PCIe Configuration](#3-operating-system--kernel-pcie-configuration)
4. [Hailo Driver & Userland Installation](#4-hailo-driver--userland-installation)
5. [Hardware Discovery & Diagnostics](#5-hardware-discovery--diagnostics)
6. [User Group & Device Permissions](#6-user-group--device-permissions)
7. [Thermal Management & Power Requirements](#7-thermal-management--power-requirements)
8. [Hardware Health Monitoring](#8-hardware-health-monitoring)

---

## 1. Hardware Architecture Overview

The target edge platform pairs the **Raspberry Pi 5** single-board computer with the official **Raspberry Pi AI HAT+** carrying the **Hailo-8L** neural processing unit:

```text
┌────────────────────────────────────────────────────────┐
│               Raspberry Pi AI HAT+                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Hailo-8L Neural Processing Unit (M.2 Key-M)     │  │
│  │  - 13 TOPS INT8 Compute Engine                   │  │
│  │  - 2.5W Typical Power Consumption                │  │
│  └────────────────────────┬─────────────────────────┘  │
└───────────────────────────┼────────────────────────────┘
                            │ 16-pin FPC PCIe Gen 3 Ribbon
┌───────────────────────────┴────────────────────────────┐
│                  Raspberry Pi 5                        │
│  - Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz     │
│  - 4GB / 8GB / 16GB LPDDR4X SDRAM                      │
│  - PCIe 2.0 / 3.0 Single-Lane Interface                │
│  - Official 27W USB-C Power Delivery                   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Physical Assembly Procedure

### Required Materials
- Raspberry Pi 5 (4GB, 8GB, or 16GB)
- Raspberry Pi AI HAT+ (with pre-installed Hailo-8L module)
- 16-pin flexible printed circuit (FPC) PCIe ribbon cable
- 4 × Brass standoffs ($M2.5 \times 16\text{mm}$) and screws
- Raspberry Pi Active Cooler (heatsink + variable speed fan)
- Official Raspberry Pi 27W USB-C Power Supply

### Assembly Steps

1. **Install the Active Cooler**:
   - Apply thermal pads to the BCM2712 SoC, RP1 I/O controller, and PMIC.
   - Align the spring-loaded push pins with the diagonal mounting holes on the Raspberry Pi 5 PCB and press down until they click securely.
   - Plug the 4-pin JST fan header into the fan socket located beside the USB-C port.

2. **Connect the PCIe Ribbon Cable to the Raspberry Pi 5**:
   - Gently pull the dark collar of the PCIe FPC connector upward to unlock it.
   - Insert the ribbon cable with the **gold contacts facing inward toward the micro-HDMI ports** and the blue pull-tab facing outward.
   - Push the locking collar down firmly and evenly on both sides.

3. **Install Standoffs**:
   - Screw the four $M2.5 \times 16\text{mm}$ brass standoffs into the corner mounting holes of the Raspberry Pi 5 from the top side.

4. **Connect Ribbon Cable to the AI HAT+**:
   - Pull open the locking collar on the underside of the AI HAT+.
   - Feed the ribbon cable through the slot and seat it firmly into the HAT+ connector with contacts facing the metal pins.
   - Engage the locking collar to secure the ribbon cable.

5. **Mount the AI HAT+**:
   - Lower the HAT+ onto the four standoffs, ensuring the GPIO header pins align smoothly.
   - Fasten the HAT+ using the four $M2.5$ screws.
   - Verify that the thermal pad on the Hailo-8L module is firmly seated against the HAT+ thermal plate.

---

## 3. Operating System & Kernel PCIe Configuration

The host operating system must be **Raspberry Pi OS (64-bit)** based on **Debian Bookworm** or later.

### Enable PCIe Gen 3 in Kernel Config

By default, the Raspberry Pi 5 operates the PCIe link at PCIe Gen 2 ($5.0\text{ GT/s}$). The Hailo-8L supports PCIe Gen 3 ($8.0\text{ GT/s}$), providing higher bandwidth:

1. Open `/boot/firmware/config.txt` in a text editor with root privileges:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```

2. Add or update the following directives under the `[all]` section:
   ```ini
   [all]
   # Enable PCIe connector
   dtparam=pciex1

   # Enable PCIe Gen 3 speeds (8.0 GT/s)
   dtparam=pciex1_gen=3
   ```

3. Save the file and reboot the system:
   ```bash
   sudo reboot
   ```

---

## 4. Hailo Driver & Userland Installation

The official Raspberry Pi OS package repositories provide pre-compiled HailoRT drivers and firmware:

```bash
# Update repository package indexes
sudo apt update

# Install Hailo kernel drivers, firmware, and CLI utilities
sudo apt install -y hailo-all
```

The `hailo-all` metapackage installs:
- `hailort`: HailoRT runtime libraries and C++/Python APIs.
- `hailortcli`: Command-line inspection and benchmarking utility.
- `hailo-firmware`: Microcode bitstreams loaded during NPU boot.
- `hailo-pci`: Dynamic Kernel Module Support (DKMS) PCIe driver.

---

## 5. Hardware Discovery & Diagnostics

Verify that the PCIe hardware link and device node are active:

### 1. Check Kernel Driver Loading
```bash
lsmod | grep hailo_pci
```
*Expected Output:*
```text
hailo_pci              65536  0
```

### 2. Verify PCI Device Discovery
```bash
lspci -nn -d 1e60:
```
*Expected Output:*
```text
0000:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor [1e60:2864] (rev 01)
```

### 3. Check System Device Node
```bash
ls -l /dev/hailo0
```
*Expected Output:*
```text
crw-rw---- 1 root hailo 242, 0 Oct  8 21:00 /dev/hailo0
```

### 4. Query HailoRT CLI Utility
```bash
hailortcli scan
```
*Expected Output:*
```text
Running Scan:
|- Device: 0000:01:00.0 (Hailo-8L)
```

---

## 6. User Group & Device Permissions

To execute HailoRT inference and compiler validation without running as root, add your user to the `hailo` and `plugdev` groups:

```bash
sudo usermod -aG hailo $USER
sudo usermod -aG plugdev $USER
```

Log out and log back in (or run `newgrp hailo`) for group membership to take effect. Test non-root access:

```bash
hailortcli fw-control identify
```

---

## 7. Thermal Management & Power Requirements

Edge neural processing generates heat under continuous high-framerate workloads:

### Power Supply
- Always power the Raspberry Pi 5 with the **official Raspberry Pi 27W USB-C Power Supply** ($5\text{V} / 5\text{A}$).
- Lower-wattage phone chargers ($15\text{W}$ or $18\text{W}$) will trigger undervoltage conditions under concurrent CPU + Hailo NPU load, causing PCIe bus disconnects or spontaneous reboots.

### Active Cooling
- The Raspberry Pi Active Cooler provides dynamic, PWM-controlled cooling.
- Under sustained YOLO11 inference, Hailo-8L core temperatures typically stabilise between $50^\circ\text{C}$ and $65^\circ\text{C}$, safely below the $85^\circ\text{C}$ throttling threshold.

---

## 8. Hardware Health Monitoring

Monitor system vitals during benchmark execution:

### Check SoC Temperature
```bash
vcgencmd measure_temp
# Example: temp=48.2'C
```

### Check Throttling and Undervoltage Flags
```bash
vcgencmd get_throttled
# Expected Output when healthy: throttled=0x0
```

**Decoding Throttling Bitmask:**
- `0x1`: Under-voltage detected right now.
- `0x2`: Arm frequency capped right now.
- `0x4`: Currently throttled.
- `0x8`: Soft temperature limit active.
- `0x10000`: Under-voltage has occurred since boot.
- `0x20000`: Arm frequency capping has occurred since boot.
- `0x40000`: Throttling has occurred since boot.
- `0x80000`: Soft temperature limit has occurred since boot.

If any undervoltage bit is active, immediately replace the power supply and cable.
