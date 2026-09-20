# Trigger a Magpie shortcut from an unattended session.
#
# Two traps in this, both of which fail silently:
#   * keybd_event is deprecated and some low-level hooks never see it;
#   * INPUT must be exactly 40 bytes on x64 (DWORD type, 4 bytes of padding so the
#     union lands on an 8-byte boundary, then the 32-byte union). A 32-byte INPUT
#     makes SendInput return 0 with ERROR_INVALID_PARAMETER and inject nothing.
#
# Magpie stores a shortcut as code | win<<8 | ctrl<<9 | alt<<10 | shift<<11, so
# 3137 (0xC41) is Alt+Shift+A and a bare F9 is 120 (0x78).
param(
    [string]$KeyList = "120",     # decimal VKs, comma separated
    [int]$Mods = 0,               # win=1 ctrl=2 alt=4 shift=8
    [int]$HoldMs = 60,
    [int]$SettleMs = 2000
)
$Keys = $KeyList -split ',' | ForEach-Object { [int] $_.Trim() }

$src = @'
using System;
using System.Runtime.InteropServices;
public static class Inp {
    [StructLayout(LayoutKind.Sequential)] public struct KEYBDINPUT {
        public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo;
    }
    [StructLayout(LayoutKind.Explicit, Size = 40)]
    public struct INPUT {
        [FieldOffset(0)] public uint type;
        [FieldOffset(8)] public KEYBDINPUT ki;
    }
    [DllImport("user32.dll", SetLastError = true)]
    public static extern uint SendInput(uint n, INPUT[] inputs, int size);
    public static uint Key(ushort vk, bool up) {
        var a = new INPUT[1];
        a[0].type = 1;                       // INPUT_KEYBOARD
        a[0].ki.wVk = vk;
        a[0].ki.dwFlags = up ? 2u : 0u;      // KEYEVENTF_KEYUP
        return SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
    }
}
'@
Add-Type -TypeDefinition $src | Out-Null

$VK_MENU = 0x12; $VK_SHIFT = 0x10; $VK_CONTROL = 0x11; $VK_LWIN = 0x5B

function Press-Mods([int]$m, [bool]$up) {
    if ($m -band 1) { [Inp]::Key($VK_LWIN, $up) | Out-Null }
    if ($m -band 2) { [Inp]::Key($VK_CONTROL, $up) | Out-Null }
    if ($m -band 4) { [Inp]::Key($VK_MENU, $up) | Out-Null }
    if ($m -band 8) { [Inp]::Key($VK_SHIFT, $up) | Out-Null }
}

foreach ($k in $Keys) {
    Press-Mods $Mods $false
    Start-Sleep -Milliseconds 50
    $r1 = [Inp]::Key([uint16]$k, $false)
    Start-Sleep -Milliseconds $HoldMs
    $r2 = [Inp]::Key([uint16]$k, $true)
    Press-Mods $Mods $true
    $e = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Output ("vk=0x{0:X2} mods=0x{1:X} down={2} up={3} err={4}" -f $k, $Mods, $r1, $r2, $e)
    Start-Sleep -Milliseconds 1500
}
Start-Sleep -Milliseconds $SettleMs
