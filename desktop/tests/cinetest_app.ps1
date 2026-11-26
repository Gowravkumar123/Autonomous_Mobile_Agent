# CineTest desktop app (WinForms) — the target for the desktop-executor tests.
# Started MINIMIZED so tests never steal focus; UI Automation patterns work regardless.
param([switch]$Visible)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$f = New-Object Windows.Forms.Form
$f.Text = "OnePiece CineTest"
$f.Size = New-Object Drawing.Size(460, 330)
$f.StartPosition = "Manual"
$f.Location = New-Object Drawing.Point(40, 40)
if (-not $Visible) { $f.WindowState = "Minimized" }

function Add-Ctl($ctl, $x, $y, $w = 140) {
    $ctl.Location = New-Object Drawing.Point($x, $y)
    $ctl.Width = $w
    $f.Controls.Add($ctl)
    return $ctl
}

$search = Add-Ctl (New-Object Windows.Forms.TextBox) 20 20 220
$search.Name = "searchBox"; $search.AccessibleName = "Search movies"
$btn = Add-Ctl (New-Object Windows.Forms.Button) 250 18 90
$btn.Text = "Search"; $btn.Name = "searchBtn"
$result = Add-Ctl (New-Object Windows.Forms.Label) 20 55 380
$result.Text = "No results"; $result.Name = "resultLabel"
$btn.Add_Click({ $result.Text = "Results for " + $search.Text + ": Scene (Telugu) 7:00 PM" })
$f.AcceptButton = $btn      # Enter in the search box submits, like a web form

$a1 = Add-Ctl (New-Object Windows.Forms.CheckBox) 20 90 120
$a1.Text = "Seat A1"; $a1.Name = "seatA1"
$a2 = Add-Ctl (New-Object Windows.Forms.CheckBox) 150 90 120
$a2.Text = "Seat A2"; $a2.Name = "seatA2"
$total = Add-Ctl (New-Object Windows.Forms.Label) 20 125 200
$total.Text = "Total Rs 0"; $total.Name = "totalLabel"
$upd = { $n = @($a1, $a2 | Where-Object { $_.Checked }).Count; $total.Text = "Total Rs " + (250 * $n) }
$a1.Add_CheckedChanged($upd); $a2.Add_CheckedChanged($upd)

$pay = Add-Ctl (New-Object Windows.Forms.Button) 20 160 160
$pay.Text = "Proceed to pay"; $pay.Name = "payBtn"
$status = Add-Ctl (New-Object Windows.Forms.Label) 20 200 380
$status.Text = "Status: selecting"; $status.Name = "statusLabel"
$pay.Add_Click({ $status.Text = "Status: PAYMENT STARTED" })
$pin = Add-Ctl (New-Object Windows.Forms.TextBox) 20 235 120
$pin.Name = "pinBox"; $pin.AccessibleName = "Card PIN"; $pin.UseSystemPasswordChar = $true

[void]$f.ShowDialog()
