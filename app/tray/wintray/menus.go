//go:build windows

package wintray

import (
	"fmt"
	"log/slog"
	"os"
	"unsafe"

	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/tray/commontray"
	"golang.org/x/sys/windows"
)

type MenuID uint32

const (
	updateAvailableMenuID MenuID = 1 + iota
	updateMenuID
	separatorMenuID
	diagLogsMenuID
	settingsMenuID
	toggleHostMenuID
	toggleDomainsMenuID
	setModelDirMenuID
	diagSeparatorMenuID
	quitMenuID
)

func (t *winTray) initMenus() error {
	if err := t.addOrUpdateMenuItem(diagLogsMenuID, 0, commontray.DiagLogsMenuTitle, false, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}
	if err := t.addOrUpdateMenuItem(settingsMenuID, 0, commontray.SettingsMenuTitle, false, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}

	setHostDisabled := false
	if os.Getenv("OLLAMA_HOST") != "" {
		setHostDisabled = true
	}
	if err := t.addOrUpdateMenuItem(toggleHostMenuID, settingsMenuID, commontray.HostMenuTitle, setHostDisabled, store.GetAllowExternalConnections()); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}
	setDomainsDisabled := false
	if os.Getenv("OLLAMA_ORIGINS") != "" {
		setDomainsDisabled = true
	}
	if err := t.addOrUpdateMenuItem(toggleDomainsMenuID, settingsMenuID, commontray.DomainMenuTitle, setDomainsDisabled, store.GetAllowBrowserConnections()); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}
	setModelsDisabled := false
	if os.Getenv("OLLAMA_MODELS") != "" {
		setModelsDisabled = true
	}
	if err := t.addOrUpdateMenuItem(setModelDirMenuID, settingsMenuID, commontray.MenuDirTitle, setModelsDisabled, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}

	if err := t.addSeparatorMenuItem(diagSeparatorMenuID, 0); err != nil {
		return fmt.Errorf("unable to create menu entries %w", err)
	}
	if err := t.addOrUpdateMenuItem(quitMenuID, 0, commontray.QuitMenuTitle, false, false); err != nil {
		return fmt.Errorf("unable to create menu entries %w\n", err)
	}
	return nil
}

func (t *winTray) UpdateAvailable(ver string) error {
	// TODO record skipping updates by specific version
	if !t.updateNotified {
		slog.Debug("updating menu and sending notification for new update")
		if err := t.addOrUpdateMenuItem(updateAvailableMenuID, 0, commontray.UpdateAvailableMenuTitle, true, false); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		if err := t.addOrUpdateMenuItem(updateMenuID, 0, commontray.UpdateMenutTitle, false, false); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		if err := t.addSeparatorMenuItem(separatorMenuID, 0); err != nil {
			return fmt.Errorf("unable to create menu entries %w", err)
		}
		iconFilePath, err := iconBytesToFilePath(wt.updateIcon)
		if err != nil {
			return fmt.Errorf("unable to write icon data to temp file: %w", err)
		}
		if err := wt.setIcon(iconFilePath); err != nil {
			return fmt.Errorf("unable to set icon: %w", err)
		}
		t.updateNotified = true

		t.pendingUpdate = true
		updateMessage, err := windows.BytePtrFromString(fmt.Sprintf(commontray.UpdateMessage, ver))
		if err != nil {
			return err
		}
		updateTitle, err := windows.BytePtrFromString("Ollama " + commontray.UpdateTitle)
		if err != nil {
			return err
		}
		// TODO - if we want to reposition the dialog down in the tray area
		// we can either build the window from scratch, or wire up
		ret, _, _ := pMessageBoxTimeout.Call(
			uintptr(t.window),
			uintptr(unsafe.Pointer(updateMessage)),
			uintptr(unsafe.Pointer(updateTitle)),
			MB_YESNO|MB_ICONINFORMATION,
			0,
			commontray.UpdateMessageTimeout*1000,
		)

		switch ret {
		case IDYES:
			select {
			case t.callbacks.Update <- struct{}{}:
			// should not happen but in case not listening
			default:
				slog.Error("no listener on Update")
			}
		case IDNO:
			// TODO - How long should we supress the pop-up?  Until next startup?  Until next version?
			slog.Info("TODO - not upgrading now by user request")
		case IDTIMEOUT:
			// TODO - depending on what we choose for NO, this may be distinct, or the same behavior
			slog.Info("TODO - user didn't press YES or NO, timeout")
		}
	}
	return nil
}
