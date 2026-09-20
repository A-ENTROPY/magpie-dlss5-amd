# Bring a window to the foreground from a background process.
#
# SetForegroundWindow alone is refused for a process that does not own the foreground,
# so the calling thread is attached to the current foreground thread first, which is the
# part that makes the call succeed.
param(
    [Parameter(Mandatory = $true)][int]$Hwnd,
    [switch]$NudgeKey            # send a short key press so the game animates
)

$src = @'
using System;
using System.Runtime.InteropServices;
public static class Fg {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool attach);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();

    [StructLayout(LayoutKind.Sequential)] public struct KEYBDINPUT {
        public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo;
    }
    [StructLayout(LayoutKind.Explicit, Size = 40)]
    public struct INPUT { [FieldOffset(0)] public uint type; [FieldOffset(8)] public KEYBDINPUT ki; }
    [DllImport("user32.dll", SetLastError = true)] public static extern uint SendInput(uint n, INPUT[] i, int s);

    public static bool Bring(IntPtr target) {
        IntPtr fg = GetForegroundWindow();
        uint fgPid = 0;
        uint fgThread = GetWindowThreadProcessId(fg, out fgPid);
        uint myThread = GetCurrentThreadId();
        bool attached = AttachThreadInput(myThread, fgThread, true);
        ShowWindow(target, 9);                 // SW_RESTORE
        bool ok = SetForegroundWindow(target);
        if (attached) AttachThreadInput(myThread, fgThread, false);
        return ok;
    }
    public static void Tap(ushort vk) {
        var a = new INPUT[1];
        a[0].type = 1; a[0].ki.wVk = vk;
        SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
        System.Threading.Thread.Sleep(400);
        a[0].ki.dwFlags = 2;                   // KEYUP
        SendInput(1, a, Marshal.SizeOf(typeof(INPUT)));
    }
}
'@
Add-Type -TypeDefinition $src | Out-Null

$h = [IntPtr] $Hwnd
$ok = [Fg]::Bring($h)
Write-Output "foregrounded=$ok hwnd=$Hwnd"
Start-Sleep -Milliseconds 800

if ($NudgeKey) {
    # W and D walk the character, which is what keeps the capture source producing frames.
    [Fg]::Tap(0x57)   # W
    [Fg]::Tap(0x44)   # D
    Write-Output "nudged"
}
