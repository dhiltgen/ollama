package wintray

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

type browseInfo struct {
	Owner       windows.Handle
	Root        uintptr // Complicated - see below
	DisplayName *uint16
	Title       *uint16
	Flags       uint32  // TBD - BIF_VALIDATE
	Callback    uintptr // callbackPtr
	Param       uintptr
	Image       int32
}

const (
	// See Windows Kits/10/Include/<version>/um/ShlObj_core.h
	BIF_RETURNONLYFSDIRS = 0x00000001
	BIF_EDITBOX          = 0x00000010
	BIF_VALIDATE         = 0x00000020
	BIF_NEWDIALOGSTYLE   = 0x00000040

	BFFM_INITIALIZED     = 1
	BFFM_SELCHANGED      = 2
	BFFM_VALIDATEFAILEDA = 3 // lParam:szPath ret:1(cont),0(EndDialog)
	BFFM_VALIDATEFAILEDW = 4 // lParam:wzPath ret:1(cont),0(EndDialog)
	BFFM_IUNKNOWN        = 5 // provides IUnknown to client. lParam: IUnknown*
	BFFM_SETSELECTION    = WM_USER + 102
	BFFM_SETEXPANDED     = WM_USER + 106
)

func BrowseForFolder(owner windows.Handle, originPath, title string) (string, error) {
	var winOrigPath [windows.MAX_PATH]uint16
	utfOrigPath, err := syscall.UTF16FromString(originPath)
	if err != nil {
		return "", err
	}
	copy(winOrigPath[:], utfOrigPath)
	cb := func(hwnd windows.Handle, msg uint32, lp, wp uintptr) uintptr {
		switch msg {
		case BFFM_INITIALIZED:
			pSendMessage.Call(
				uintptr(hwnd),
				BFFM_SETSELECTION,
				1,
				uintptr(unsafe.Pointer(&winOrigPath[0])),
			)
			pSendMessage.Call(
				uintptr(hwnd),
				BFFM_SETEXPANDED,
				1,
				uintptr(unsafe.Pointer(&winOrigPath[0])),
			)
			// default:
			// 	slog.Debug("XXX directory browse callback called", "msg", msg, "lp", lp, "wp", wp)
		}
		return 0
	}
	cbPtr := syscall.NewCallback(cb)
	ret, _, err := pOleInitialize.Call(0)
	if ret != 0 {
		return "", fmt.Errorf("ole initialize failure: %d %w", ret, err)
	}
	defer pOleUninitialize.Call(0)

	var dispName [windows.MAX_PATH]uint16

	utfTitle, err := syscall.UTF16FromString(title)
	if err != nil {
		return "", err
	}

	bi := browseInfo{
		Owner:       owner,
		DisplayName: &dispName[0],
		Title:       &utfTitle[0],
		Flags:       BIF_RETURNONLYFSDIRS | BIF_EDITBOX | BIF_VALIDATE | BIF_NEWDIALOGSTYLE,
		Callback:    cbPtr,
	}
	var buf [windows.MAX_PATH]uint16
	copy(buf[:], utfOrigPath)

	pidl, _, _ := pSHBrowseForFolderW.Call(uintptr(unsafe.Pointer(&bi)))
	if pidl == 0 {
		// User pressed cancel
		return originPath, nil
	}
	defer pCoTaskMemFree.Call(pidl)

	var path [windows.MAX_PATH]uint16
	ret, _, err = pSHGetPathFromIDListW.Call(
		pidl,
		uintptr(unsafe.Pointer(&path[0])),
	)
	if ret != 1 {
		return "", err
	}

	location := syscall.UTF16ToString(path[:])
	return location, nil
}
