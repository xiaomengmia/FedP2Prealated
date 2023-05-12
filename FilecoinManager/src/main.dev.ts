/* eslint global-require: off, no-console: off */

/**
 * This module executes inside of electron's main process. You can start
 * electron renderer process from here and communicate with the other processes
 * through IPC.
 *
 * When running `yarn build` or `yarn build:main`, this file is compiled to
 * `./src/main.prod.js` using webpack. This gives us some performance wins.
 */
import 'core-js/stable';
import 'regenerator-runtime/runtime';
import { app, BrowserWindow, shell } from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import * as os from 'os';
import MenuBuilder from './menu';
// import path from 'path';
const path = require('path');

const { ipcMain, dialog } = require('electron');

export default class AppUpdater {
  constructor() {
    log.transports.file.level = 'info';
    autoUpdater.logger = log;
    autoUpdater.checkForUpdatesAndNotify();
  }
}

let mainWindow: BrowserWindow | null = null;

if (process.env.NODE_ENV === 'production') {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

if (
  process.env.NODE_ENV === 'development' ||
  process.env.DEBUG_PROD === 'true'
) {
  require('electron-debug')();
}

const installExtensions = async () => {
  const installer = require('electron-devtools-installer');
  const forceDownload = !!process.env.UPGRADE_EXTENSIONS;
  const extensions = ['REACT_DEVELOPER_TOOLS'];

  return installer
    .default(
      extensions.map((name) => installer[name]),
      forceDownload
    )
    .catch(console.log);
};

const createWindow = async () => {
  if (
    process.env.NODE_ENV === 'development' ||
    process.env.DEBUG_PROD === 'true'
  ) {
    await installExtensions();
  }

  const RESOURCES_PATH = app.isPackaged
    ? path.join(process.resourcesPath, 'assets')
    : path.join(__dirname, '../assets');

  const getAssetPath = (...paths: string[]): string => {
    return path.join(RESOURCES_PATH, ...paths);
  };

  mainWindow = new BrowserWindow({
    show: false,
    width: 1024,
    height: 728,
    icon: getAssetPath('icon.png'),
    webPreferences: {
      nodeIntegration: true,
    },
    frame: false,
  });

  mainWindow.loadURL(`file://${__dirname}/index.html`);

  // @TODO: Use 'ready-to-show' event
  //        https://github.com/electron/electron/blob/master/docs/api/browser-window.md#using-ready-to-show-event
  mainWindow.webContents.on('did-finish-load', () => {
    if (!mainWindow) {
      throw new Error('"mainWindow" is not defined');
    }
    if (process.env.START_MINIMIZED) {
      mainWindow.minimize();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  const menuBuilder = new MenuBuilder(mainWindow);
  menuBuilder.buildMenu();

  // Open urls in the user's browser
  mainWindow.webContents.on('new-window', (event, url) => {
    event.preventDefault();
    shell.openExternal(url);
  });

  // Remove this if your app does not use auto updates
  // eslint-disable-next-line
  new AppUpdater();
};

/**
 * Add event listeners...
 */

app.on('window-all-closed', () => {
  // Respect the OSX convention of having the application in memory even
  // after all windows have been closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.whenReady().then(createWindow).catch(console.log);

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) createWindow();
});

const fs = require('fs');

ipcMain.on('open-folder-dialog', async (event: { returnValue: any }) => {
  await dialog.showOpenDialog(
    {
      properties: ['openDirectory', 'openFile'],
    },
    (files: any) => {
      if (files) {
        event.returnValue = files;
      }
    }
  );
});
ipcMain.handle('open-file-dialog', (event) => {
  const files = dialog.showOpenDialogSync({
    properties: ['openFile', 'multiSelections'],
  });
  if (!files) {
    return {};
  }
  const filesSizes = files.map((path) => {
    return fs.statSync(path).size;
  });
  return { files, filesSizes };
});

const finalFiles: any[] = [];

function fileDisplay(filePath) {
  fs.readdir(filePath, function (err, files) {
    if (err) {
      console.warn(err);
    } else {
      files.forEach(function (filename) {
        const filedir = path.join(filePath, filename);
        fs.stat(filedir, function (eror, stats) {
          if (eror) {
            // console.warn('');
          } else {
            const isFile = stats.isFile();
            const isDir = stats.isDirectory();
            if (isFile) {
              finalFiles.push(filedir);
            }
            if (isDir) {
              fileDisplay(filedir);
            }
          }
        });
      });
    }
  });
}
const glob = require('glob');

const getFiles = function (src) {
  return glob.sync(`${src}/**/*`);
};

const getFolders = function (src) {
  return glob.sync(`${src}/**/*/`);
};

ipcMain.handle('open-folder-dialog', (event) => {
  const files = dialog.showOpenDialogSync({
    properties: ['openDirectory'],
  });
  if (files) {
    const filePath = path.resolve(files[0]);
    const filesInPath = getFiles(filePath);
    const filesSizes = filesInPath.map((path) => {
      return fs.statSync(path).size;
    });
    return { filePath, filesInPath, filesSizes };
  }
  // const folders = getFolders(filePath);
  return {};
});

ipcMain.handle('list-folder', (event, folderPath) => {
  // const files = dialog.showOpenDialogSync({
  //   properties: ['openDirectory'],
  // });
  if (!folderPath) {
    return {};
  }
  const filesInPath = getFiles(folderPath);

  const filesSizes = filesInPath.map((path) => {
    return fs.statSync(path).size;
  });
  return { filesInPath, filesSizes };
});
