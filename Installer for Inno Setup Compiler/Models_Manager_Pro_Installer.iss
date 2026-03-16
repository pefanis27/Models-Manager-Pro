; Models_Manager_Pro_Installer.iss  (Improved / Hardened)
; ------------------------------------------------------------
; Installer για Models Manager Pro A.I Copilot
; ✅ Υποστηρίζει onedir ή onefile (διάλεξε με το BuildMode παρακάτω)
; ✅ Χωρίς hard-coded absolute paths (δουλεύει όπου κι αν βρίσκεται το project)
; ✅ Δεν πειράζει το PATH των Windows (DLL paths λύνονται από την εφαρμογή + runtime hook)
; ✅ Best-practice: PrivilegesRequired=lowest (χωρίς admin), καλύτερο upgrade behavior
; ✅ Προαιρετικό Desktop icon (Tasks)
; ✅ Κλείνει το app αν τρέχει (CloseApplications)
; ✅ Πιο "ανθεκτικό" σε απουσία app_icon.ico / installer_info.txt (με preprocessor checks)
; ------------------------------------------------------------

#define MyAppName "Models Manager Pro A.I Copilot"
#define MyAppVersion "4.0"
#define MyAppPublisher "Ευάγγελος Πεφάνης"
#define MyAppExeName "Models_Manager_Pro.exe"
#define MyAppIdGuid "B3F0C47B-1234-4567-89AB-ABCDEF123456"

; ------------------------------------------------------------
; ΕΠΙΛΟΓΗ BUILD MODE:
;   "onedir"  -> dist\Models_Manager_Pro\*   (προτεινόμενο για CUDA)
;   "onefile" -> dist\Models_Manager_Pro.exe
; ------------------------------------------------------------
#define BuildMode "onedir"

#define SourceRoot SourcePath
#if BuildMode == "onedir"
  #define DistRoot SourceRoot + "\\dist\\Models_Manager_Pro"
#else
  #define DistRoot SourceRoot + "\\dist"
#endif

#define TensorRTDir SourcePath + "TensorRT-10.13.3.9"
#define DataSetsDir SourcePath + "Data_Sets"

; --- REQUIRED folders (fail compile if missing) ---
#if DirExists(TensorRTDir) == 0
  #error TensorRT folder not found: {#TensorRTDir}
#endif
#if DirExists(DataSetsDir) == 0
  #error Data_Sets folder not found: {#DataSetsDir}
#endif


; Optional setup resources
#define SetupIconMaybe SourceRoot + "\\app_icon.ico"
#define InfoBeforeMaybe SourceRoot + "\\installer_info.txt"

[Setup]
; Σταθερό AppId ώστε installer/update να "βλέπουν" την ίδια εφαρμογή
AppId={{{#MyAppIdGuid}}}

AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}

AppPublisher={#MyAppPublisher}
AppPublisherURL=https://www.ece.uop.gr/staff/pefanis-evangelos/
AppSupportURL=https://drive.google.com/drive/folders/1XbFDVJabpz6fyNt2Jy-Dx4LHNUcguN8z?usp=sharing
AppUpdatesURL=https://drive.google.com/drive/folders/1MAYXibQEpMI-nnQ4TmaqIEZN_5eiKYco?usp=sharing
AppComments=Εφαρμογή διαχείρισης, εκπαίδευσης και αξιολόγησης μοντέλων.

; Non-admin install (προτεινόμενο για ML apps με πολλαπλά DLLs & user data)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

DefaultDirName={%USERPROFILE}\Models_Manager_Pro_A.I_Copilot
DefaultGroupName={#MyAppName}

DisableDirPage=no
DisableProgramGroupPage=no
UsePreviousAppDir=yes

OutputBaseFilename=Models_Manager_Pro_A.I_Copilot_Setup
OutputDir=.
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

; Κλείνει το app αν τρέχει (αποφεύγει "file in use")
CloseApplications=yes
CloseApplicationsFilter={#MyAppExeName}
RestartApplications=no

; Καθαρότερη εμφάνιση στο Programs & Features
UninstallDisplayIcon={app}\{#MyAppExeName}

; Optional setup icon / info before (compile-safe)
#if FileExists(SetupIconMaybe)
SetupIconFile={#SetupIconMaybe}
#endif

#if FileExists(InfoBeforeMaybe)
InfoBeforeFile={#InfoBeforeMaybe}
#endif

[Languages]
Name: "greek"; MessagesFile: "compiler:Languages\Greek.isl"

[Tasks]
Name: "desktopicon"; Description: "Δημιουργία εικονιδίου στην επιφάνεια εργασίας"; Flags: unchecked

[Dirs]
; Δημιουργούμε τους βασικούς φακέλους (η εφαρμογή έτσι κι αλλιώς τους ξανα-εξασφαλίζει)
Name: "{app}\Data_Sets"; Flags: uninsneveruninstall
Name: "{app}\Trained_Models"; Flags: uninsneveruninstall
Name: "{app}\Crash_Logs"; Flags: uninsneveruninstall

[Files]
; ------------------ Required: TensorRT folder ------------------
Source: "{#TensorRTDir}\*"; DestDir: "{app}\TensorRT-10.13.3.9"; Flags: ignoreversion recursesubdirs createallsubdirs

; ------------------ Required: Data_Sets (kept on uninstall) ------------------
Source: "{#DataSetsDir}\*"; DestDir: "{app}\Data_Sets"; Flags: ignoreversion recursesubdirs createallsubdirs uninsneveruninstall

; ------------------ App files ------------------
#if BuildMode == "onedir"
Source: "{#DistRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "{#DistRoot}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
#endif

; ------------------ Optional: TensorRT folder (αν υπάρχει) ------------------

; ------------------ Optional: Data_Sets (αν υπάρχει τοπικά) ------------------

[Icons]
Name: "{autoprograms}\{#MyAppName}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Εκκίνηση του {#MyAppName}"; Flags: nowait postinstall skipifsilent
