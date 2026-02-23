[Setup]
AppName="Reprojection toolbox"
AppVersion=1.0
DefaultDirName={localappdata}\Reprojection_toolbox
DisableProgramGroupPage=yes
Compression=lzma2
SolidCompression=yes
OutputDir=.
OutputBaseFilename=RT_Setup
PrivilegesRequired=lowest

[Files]
Source: "installer.bat"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "metashape-2.3.0-cp39.cp310.cp311.cp312.cp313-none-win_amd64.whl"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "RT.zip"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "Logo-Ifremer.ico"; DestDir: "{app}";

[Run]
Filename: "{tmp}\installer.bat"; WorkingDir: "{tmp}"; StatusMsg: "Installation de Reprojection Toolbox en cours...";

[Tasks]
Name: "desktopicon"; Description: "Créer une icône sur le Bureau"; GroupDescription: "Raccourcis:"

[Dirs]
Name: "{userprograms}\Reprojection_toolbox"

[Icons]
Name: "{userprograms}\Reprojection_toolbox\Reprojection_toolbox"; Filename: "{localappdata}\Reprojection_toolbox\RT.bat"; IconFilename: "{app}\Logo-Ifremer.ico"
Name: "{commondesktop}\Reprojection_toolbox"; Filename: "{localappdata}\Reprojection_toolbox\RT.bat"; Tasks: desktopicon; IconFilename: "{app}\Logo-Ifremer.ico"
