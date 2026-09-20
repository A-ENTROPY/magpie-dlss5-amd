# Drive one Magpie scaling session for unattended testing.
#
# Magpie's scale shortcut is stored as code | win<<8 | ctrl<<9 | alt<<10 | shift<<11,
# so 3137 (0xC41) is Alt+Shift+A. SendKeys cannot drive a RegisterHotKey registration
# because it does not produce real key events, so this injects them.
param(
    [int]$Key = 41,        # 'A'
    [switch]$WithAlt = $true,
    [switch]$WithShift = $true,
    [int]$SettleMs = 1500
)

$sig = @'
[DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
'@
$kb = Add-Type -MemberDefinition $sig -Name Kb -Namespace W32 -PassThru

$KEYUP = 2
$VK_MENU = 0x12
$VK_SHIFT = 0x10

if ($WithAlt)   { $kb::keybd_event($VK_MENU,  0, 0, [UIntPtr]::Zero) }
if ($WithShift) { $kb::keybd_event($VK_SHIFT, 0, 0, [UIntPtr]::Zero) }
Start-Sleep -Milliseconds 60
$kb::keybd_event([byte]$Key, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 60
$kb::keybd_event([byte]$Key, 0, $KEYUP, [UIntPtr]::Zero)
if ($WithShift) { $kb::keybd_event($VK_SHIFT, 0, $KEYUP, [UIntPtr]::Zero) }
if ($WithAlt)   { $kb::keybd_event($VK_MENU,  0, $KEYUP, [UIntPtr]::Zero) }

Write-Output "sent vk=0x$('{0:X2}' -f $Key) alt=$WithAlt shift=$WithShift"
Start-Sleep -Milliseconds $SettleMs
