import {
  ChonkyActions,
  ChonkyFileActionData,
  FileArray,
  FileBrowserProps,
  FileData,
  FileHelper,
  FullFileBrowser,
  defineFileAction,
  ChonkyIconName,
  FileBrowser,
  FileContextMenu,
  FileList,
  FileNavbar,
  FileToolbar,
} from 'chonky';
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import SlidingPanel, { PanelType } from 'react-sliding-side-panel';

import clsx from 'clsx';

// import { Tabs, Tab, Typography, Box, Button } from '@material-ui/core';
import {
  createStyles,
  makeStyles,
  Theme,
  withStyles,
} from '@material-ui/core/styles';

import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Avatar from '@material-ui/core/Avatar';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import ListItemAvatar from '@material-ui/core/ListItemAvatar';

import ImageIcon from '@material-ui/icons/Image';
import WorkIcon from '@material-ui/icons/Work';
import BeachAccessIcon from '@material-ui/icons/BeachAccess';
import WifiIcon from '@material-ui/icons/Wifi';
import BluetoothIcon from '@material-ui/icons/Bluetooth';
import FolderIcon from '@material-ui/icons/Folder';
import DeleteIcon from '@material-ui/icons/Delete';
import RestoreIcon from '@material-ui/icons/Restore';
import GetAppIcon from '@material-ui/icons/GetApp';
import StorageIcon from '@material-ui/icons/Storage';
import DescriptionIcon from '@material-ui/icons/Description';

import DialogTitle from '@material-ui/core/DialogTitle';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import PersonIcon from '@material-ui/icons/Person';
import AddIcon from '@material-ui/icons/Add';
import Typography from '@material-ui/core/Typography';
import { blue } from '@material-ui/core/colors';
import Grid from '@material-ui/core/Grid';
import Slider from '@material-ui/core/Slider';
import Input from '@material-ui/core/Input';

import {
  CircularProgress,
  Divider,
  IconButton,
  ListItemIcon,
  ListItemSecondaryAction,
  Paper,
  Snackbar,
  Step,
  StepConnector,
  StepIconProps,
  StepLabel,
  Stepper,
  Switch,
} from '@material-ui/core';
import { Check } from '@material-ui/icons';
import DemoFsMap from './fs_map.json';

const path = require('path');

const { ipcRenderer } = window.require('electron');
// We define a custom file data because our File object has more properties
interface CustomFileData extends FileData {
  parentId?: string;
  childrenIds?: string[];
  modDates?: Date[];
  isSyncDir?: boolean;
  localPath?: string;
}

interface CustomFileMap {
  [fileId: string]: CustomFileData;
}

// Helper method to attach our custom TypeScript types to the imported JSON file map.
const prepareCustomFileMap = () => {
  const baseFileMap = (DemoFsMap.fileMap as unknown) as CustomFileMap;
  const { rootFolderId } = DemoFsMap;
  return { baseFileMap, rootFolderId };
};

const useCustomFileMap = () => {
  const { baseFileMap, rootFolderId } = useMemo(prepareCustomFileMap, []);

  // Setup the React state for our file map and the current folder.
  const [fileMap, setFileMap] = useState(baseFileMap);
  const [currentFolderId, setCurrentFolderId] = useState(rootFolderId);

  // Setup the function used to reset our file map to its initial value
  const resetFileMap = useCallback(() => {
    setFileMap(baseFileMap);
    setCurrentFolderId(rootFolderId);
  }, [baseFileMap, rootFolderId]);

  const currentFolderIdRef = useRef(currentFolderId);
  useEffect(() => {
    currentFolderIdRef.current = currentFolderId;
  }, [currentFolderId]);

  // Function that will be called when user deletes files either using the toolbar
  // button or Delete key.
  const deleteFiles = useCallback((files: CustomFileData[]) => {
    setFileMap((currentFileMap) => {
      const newFileMap = { ...currentFileMap };

      files.forEach((file) => {
        // Delete file from the file map.
        delete newFileMap[file.id];

        // Update the parent folder
        if (file.parentId) {
          const parent = newFileMap[file.parentId]!;
          const newChildrenIds = parent.childrenIds!.filter(
            (id) => id !== file.id
          );
          newFileMap[file.parentId] = {
            ...parent,
            childrenIds: newChildrenIds,
            childrenCount: newChildrenIds.length,
          };
        }
      });

      return newFileMap;
    });
  }, []);

  // drag & drop functions
  const moveFiles = useCallback(
    (
      files: CustomFileData[],
      source: CustomFileData,
      destination: CustomFileData
    ) => {
      setFileMap((currentFileMap) => {
        const newFileMap = { ...currentFileMap };
        const moveFileIds = new Set(files.map((f) => f.id));

        // Delete old files
        const newSourceChildrenIds = source.childrenIds!.filter(
          (id) => !moveFileIds.has(id)
        );
        newFileMap[source.id] = {
          ...source,
          childrenIds: newSourceChildrenIds,
          childrenCount: newSourceChildrenIds.length,
        };

        // Add the new files
        const newDestinationChildrenIds = [
          ...destination.childrenIds!,
          ...files.map((f) => f.id),
        ];
        newFileMap[destination.id] = {
          ...destination,
          childrenIds: newDestinationChildrenIds,
          childrenCount: newDestinationChildrenIds.length,
        };

        // Finally, update the parent folder ID on the files from source folder
        // ID to the destination folder ID.
        files.forEach((file) => {
          newFileMap[file.id] = {
            ...file,
            parentId: destination.id,
          };
        });

        return newFileMap;
      });
    },
    []
  );

  // Function that will be called when user creates a new folder using the toolbar
  // button.
  const idCounter = useRef(0);
  const createFolder = useCallback((folderName: string) => {
    setFileMap((currentFileMap) => {
      const newFileMap = { ...currentFileMap };

      // Create the new folder
      const newFolderId = `new-folder-${idCounter.current++}`;
      newFileMap[newFolderId] = {
        id: newFolderId,
        name: folderName,
        isDir: true,
        modDate: new Date(),
        parentId: currentFolderIdRef.current,
        childrenIds: [],
        childrenCount: 0,
      };

      // Update parent folder to reference the new folder.
      const parent = newFileMap[currentFolderIdRef.current];
      newFileMap[currentFolderIdRef.current] = {
        ...parent,
        childrenIds: [...parent.childrenIds!, newFolderId],
      };

      return newFileMap;
    });
  }, []);

  const createFile = useCallback(
    (folderName: string, size?: number, folderId?: string) => {
      setFileMap((currentFileMap) => {
        const newFileMap = { ...currentFileMap };
        let parent;
        if (folderId) {
          parent = newFileMap[folderId];
        } else {
          parent = newFileMap[currentFolderIdRef.current];
        }
        const oldFileId = parent.childrenIds!.find(
          (id) => folderName === newFileMap[id].name
        );
        if (oldFileId) {
          if (newFileMap[oldFileId].modDates !== undefined) {
            newFileMap[oldFileId] = {
              ...newFileMap[oldFileId],
              modDates: [...newFileMap[oldFileId].modDates!, new Date()],
            };
          }
        } else {
          // Create the new folder
          const newFolderId = `new-folder-${idCounter.current++}`;
          newFileMap[newFolderId] = {
            id: newFolderId,
            name: folderName,
            modDate: new Date(),
            modDates: [new Date()],
            parentId: currentFolderIdRef.current,
            size,
          };
          // Update parent folder to reference the new folder.

          newFileMap[currentFolderIdRef.current] = {
            ...parent,
            childrenIds: [...parent.childrenIds!, newFolderId],
          };
        }

        return newFileMap;
      });
    },
    []
  );

  const createSyncFolder = useCallback(
    (folderPath: string, files: string[], filesSizes: number[]) => {
      setFileMap((currentFileMap) => {
        const newFileMap = { ...currentFileMap };

        // Create the new folder
        const newFolderId = `new-folder-${idCounter.current++}`;
        newFileMap[newFolderId] = {
          id: newFolderId,
          name: path.basename(folderPath),
          isDir: true,
          modDate: new Date(),
          parentId: currentFolderIdRef.current,
          childrenIds: [],
          childrenCount: 0,
          isSyncDir: true,
          color: '#00d9ff',
          localPath: folderPath,
        };

        // Update parent folder to reference the new folder.
        let parent = newFileMap[currentFolderIdRef.current];
        newFileMap[currentFolderIdRef.current] = {
          ...parent,
          childrenIds: [...parent.childrenIds!, newFolderId],
        };

        if (files) {
          parent = newFileMap[newFolderId];
          files.forEach((file, i) => {
            const fileName: string = path.basename(file);

            const oldFileId = parent.childrenIds!.find(
              (id) => fileName === newFileMap[id].name
            );
            if (oldFileId) {
              if (newFileMap[oldFileId].modDates !== undefined) {
                newFileMap[oldFileId] = {
                  ...newFileMap[oldFileId],
                  modDates: [...newFileMap[oldFileId].modDates!, new Date()],
                };
              }
            } else {
              const newFileId = `new-file-${idCounter.current++}`;
              newFileMap[newFileId] = {
                id: newFileId,
                name: fileName,
                modDate: new Date(),
                modDates: [new Date()],
                parentId: newFolderId,
                icon: ChonkyIconName.upload,
                size: filesSizes[i],
              };
              newFileMap[newFolderId] = {
                ...newFileMap[newFolderId],
                childrenIds: [
                  ...newFileMap[newFolderId].childrenIds!,
                  newFileId,
                ],
              };
            }
          });
        }
        return newFileMap;
      });
    },
    []
  );

  const syncFile = useCallback(() => {
    setFileMap((currentFileMap) => {
      const newFileMap = { ...currentFileMap };
      const parent = newFileMap[currentFolderIdRef.current];

      parent.childrenIds!.forEach((id) => {
        if (!newFileMap[id].isDir) {
          newFileMap[id] = {
            ...newFileMap[id],
            icon: ChonkyIconName.file,
          };
        }
      });


      return newFileMap;
    });
  }, []);

  const refreshLocalFolder = useCallback(() => {
    setFileMap((currentFileMap) => {
      const newFileMap = { ...currentFileMap };
      const parent = newFileMap[currentFolderIdRef.current];
      ipcRenderer.invoke('list-folder').then(({ newFiles, newSizes }) => {
        newFiles.forEach((file, i) => {
          const fileName: string = path.basename(file);
          let updateId: number = parent.childrenIds!.find((id) => {
            return (
              newFileMap[id].name === fileName &&
              newFileMap[id].size !== newSizes[i]
            );
          });
          if (updateId) {
            newFiles[updateId] = {
              ...newFiles[updateId],
              size: newSizes[i],
              icon: ChonkyIconName.upload,
            };
          } else {
            updateId = parent.childrenIds!.find((id) => {
              return newFileMap[id].name === fileName;
            });
            if (!updateId) {
              const newFileId = `new-file-${idCounter.current++}`;
              newFileMap[newFileId] = {
                id: newFileId,
                name: fileName,
                modDate: new Date(),
                modDates: [new Date()],
                parentId: currentFolderIdRef.current,
                icon: ChonkyIconName.upload,
                size: newSizes[i],
              };
              // Update parent folder to reference the new folder.

              newFileMap[currentFolderIdRef.current] = {
                ...newFileMap[currentFolderIdRef.current],
                childrenIds: [
                  ...newFileMap[currentFolderIdRef.current].childrenIds!,
                  newFileId,
                ],
              };
            }
          }
        });

        return newFileMap;
      });
    });
  }, []);

  return {
    fileMap,
    currentFolderId,
    setCurrentFolderId,
    resetFileMap,
    deleteFiles,
    moveFiles,
    createFolder,
    createFile,
    createSyncFolder,
    syncFile,
    refreshLocalFolder,
  };
};

export const useFiles = (
  fileMap: CustomFileMap,
  currentFolderId: string
): FileArray => {
  return useMemo(() => {
    const currentFolder = fileMap[currentFolderId];
    const childrenIds = currentFolder.childrenIds!;
    const files = childrenIds.map((fileId: string) => fileMap[fileId]);
    return files;
  }, [currentFolderId, fileMap]);
};

export const useFolderChain = (
  fileMap: CustomFileMap,
  currentFolderId: string
): FileArray => {
  return useMemo(() => {
    const currentFolder = fileMap[currentFolderId];

    const folderChain = [currentFolder];

    let { parentId } = currentFolder;
    while (parentId) {
      const parentFile = fileMap[parentId];
      if (parentFile) {
        folderChain.unshift(parentFile);
        parentId = parentFile.parentId;
      } else {
        break;
      }
    }

    return folderChain;
  }, [currentFolderId, fileMap]);
};

const AddSyncFolder = defineFileAction({
  id: 'add_sync_folder',
  // sortKeySelector: (file: Nullable<FileData>) => (file ? file.size : undefined),
  button: {
    name: 'Add Sync Folder',
    toolbar: true,
    group: 'Sync',
  },
} as const);

const SynchronizeFolder = defineFileAction({
  id: 'sync_folder',
  // sortKeySelector: (file: Nullable<FileData>) => (file ? file.size : undefined),
  button: {
    name: 'Synchronize',
    toolbar: true,
    group: 'Sync',
  },
} as const);

const RefreshLocalFolder = defineFileAction({
  id: 'refresh_local_folder',
  // sortKeySelector: (file: Nullable<FileData>) => (file ? file.size : undefined),
  button: {
    name: 'Refresh',
    toolbar: true,
    group: 'Sync',
  },
} as const);

const Properties = defineFileAction({
  id: 'properties',
  requiresSelection: true,
  fileFilter: (file) => file && !file.isDir,
  button: {
    name: 'Properties',
    toolbar: true,
    contextMenu: true,
  },
} as const);

export const useFileActionHandler = (
  setCurrentFolderId: (folderId: string) => void,
  deleteFiles: (files: CustomFileData[]) => void,
  moveFiles: (
    files: FileData[],
    source: FileData,
    destination: FileData
  ) => void,
  createFolder: (folderName: string) => void,
  createFile: (folderName: string, size?: number, folderId?: string) => void,
  createSyncFolder: (
    folderName: string,
    files: string[],
    filesSizes: any[]
  ) => void,
  syncFile,
  refreshLocalFolder,
  handleClickOpen: (fileData: CustomFileData) => void
) => {
  return useCallback(
    async (data: ChonkyFileActionData) => {
      if (data.id === ChonkyActions.OpenFiles.id) {
        const { targetFile, files } = data.payload;
        const fileToOpen = targetFile ?? files[0];
        if (fileToOpen && FileHelper.isDirectory(fileToOpen)) {
          setCurrentFolderId(fileToOpen.id);
        }
      } else if (data.id === ChonkyActions.DeleteFiles.id) {
        deleteFiles(data.state.selectedFilesForAction!);
      } else if (data.id === ChonkyActions.MoveFiles.id) {
        moveFiles(
          data.payload.files,
          data.payload.source!,
          data.payload.destination
        );
      } else if (data.id === ChonkyActions.CreateFolder.id) {
        // const folderName = prompt('Provide the name for your new folder:');
        // if (folderName) createFolder(folderName);
        // const fileName = await ipcRenderer.sendSync('open-folder-dialog');
        // const folderName: string = path.basename(path.dirname(file));
        ipcRenderer.invoke('open-folder-dialog').then((files) => {
          if (files) {
            files.forEach((file) => {
              if (file.slice(-1) === '\\') {
                const folderName: string = path.basename(path.dirname(file));
                if (folderName) createFolder(folderName);
              } else {
                const folderName: string = path.basename(file);
                if (folderName) createFile(folderName);
              }
            });
          }
        });
      } else if (data.id === ChonkyActions.UploadFiles.id) {
        // const folderName = prompt('Provide the name for your new file:');
        // if (folderName) createFile(folderName);
        ipcRenderer.invoke('open-file-dialog').then(({ files, filesSizes }) => {
          if (files) {
            files.forEach((file, index) => {
              const folderName: string = path.basename(file);
              if (folderName) createFile(folderName, filesSizes[index]);
            });
          }
        });
      } else if (data.id === AddSyncFolder.id) {
        ipcRenderer
          .invoke('open-folder-dialog')
          .then(({ filePath, filesInPath, filesSizes }) => {
            createSyncFolder(filePath, filesInPath, filesSizes);
            // createFolder(path.basename(filePath));
            // if (filesInPath) {
            //   filesInPath.forEach((file) => {
            //     const folderName: string = path.basename(file);
            //     if (folderName) createFile(folderName);
            //   });
            // }
          });
      } else if (data.id === SynchronizeFolder.id) {
        syncFile();
      } else if (data.id === RefreshLocalFolder.id) {
        refreshLocalFolder();
      } else if (data.id === Properties.id) {
        handleClickOpen(
          data.state.selectedFiles ? data.state.selectedFiles![0] : undefined
        );
      }

      // showActionNotification(data);
    },
    [createFile, createFolder, deleteFiles, moveFiles, setCurrentFolderId]
  );
};

export type VFSProps = Partial<FileBrowserProps>;

const QontoConnector = withStyles({
  alternativeLabel: {
    top: 10,
    left: 'calc(-50% + 16px)',
    right: 'calc(50% + 16px)',
  },
  active: {
    '& $line': {
      borderColor: '#09f',
    },
  },
  completed: {
    '& $line': {
      borderColor: '#09f',
    },
  },
  line: {
    borderColor: '#eaeaf0',
    borderTopWidth: 3,
    borderRadius: 1,
  },
})(StepConnector);

const useQontoStepIconStyles = makeStyles({
  root: {
    color: '#eaeaf0',
    display: 'flex',
    height: 22,
    alignItems: 'center',
  },
  active: {
    color: '#09f',
  },
  circle: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    backgroundColor: 'currentColor',
  },
  completed: {
    color: '#09f',
    zIndex: 1,
    fontSize: 18,
  },
});

function QontoStepIcon(props: StepIconProps) {
  const classes = useQontoStepIconStyles();
  const { active, completed } = props;

  return (
    <div
      className={clsx(classes.root, {
        [classes.active]: active,
      })}
    >
      {completed ? (
        <Check className={classes.completed} />
      ) : (
        <CircularProgress size={10} />
      )}
    </div>
  );
}

function getSteps() {
  return [
    'Client Funding',
    'Check For Acceptance',
    'Awaiting Pre Commit',
    'Sealing',
  ];
}

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      width: '100%',
      maxWidth: 800,
      backgroundColor: theme.palette.background.paper,
    },
    input: {
      width: 42,
    },
  })
);

export interface SimpleDialogProps {
  open: boolean;
  selectedValue: CustomFileData;
  onClose: (value: string) => void;
}

function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

function SimpleDialog(props: SimpleDialogProps) {
  const classes = useStyles();
  const { onClose, selectedValue, open } = props;

  const handleClose = () => {
    onClose(selectedValue);
  };

  const handleListItemClick = (value: string) => {
    onClose(value);
  };

  const [checked, setChecked] = React.useState(['wifi', 'cold']);

  const handleToggle = (value: string) => () => {
    const currentIndex = checked.indexOf(value);
    const newChecked = [...checked];

    if (currentIndex === -1) {
      newChecked.push(value);
    } else {
      newChecked.splice(currentIndex, 1);
    }

    setChecked(newChecked);
  };

  const [value, setValue] = React.useState<
    number | string | Array<number | string>
  >(1);

  const [dealMinDuration, setDealMinDuration] = React.useState<
    number | string | Array<number | string>
  >(1000);

  const [maxPrice, setmaxPrice] = React.useState<
    number | string | Array<number | string>
  >(1000);

  const handleSliderChange = (event: any, newValue: number | number[]) => {
    setValue(newValue);
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setValue(event.target.value === '' ? '' : Number(event.target.value));
  };

  const handleInputChangeDealMinDuration = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setDealMinDuration(
      event.target.value === '' ? '' : Number(event.target.value)
    );
  };

  const handleInputChangeMaxPrice = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setmaxPrice(event.target.value === '' ? '' : Number(event.target.value));
  };

  const handleBlur = () => {
    if (value < 0) {
      setValue(0);
    } else if (value > 100) {
      setValue(100);
    }
  };

  const [activeStep, setActiveStep] = React.useState(0);
  const steps = getSteps();

  const [progress, setProgress] = React.useState(10);

  React.useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prevProgress) =>
        prevProgress >= 100 ? 10 : prevProgress + 10
      );
    }, 800);
    return () => {
      clearInterval(timer);
    };
  }, []);

  // @ts-ignore
  return (
    <Dialog
      onClose={handleClose}
      aria-labelledby="simple-dialog-title"
      open={open}
      fullWidth
      maxWidth="sm"
      scroll="paper"
    >
      <DialogContent dividers>
        <DialogTitle id="form-dialog-title">File Properties</DialogTitle>
        <List className={classes.root}>
          <ListItem>
            <ListItemAvatar>
              <Avatar>
                <DescriptionIcon />
              </Avatar>
            </ListItemAvatar>
            <ListItemText
              primary={selectedValue ? selectedValue.name : 'empty'}
              secondary={selectedValue ? formatBytes(selectedValue.size) : '0'}
            />
          </ListItem>
          <Divider component="li" />
          {selectedValue
            ? selectedValue.modDates?.map((date) => {
                return (
                  <ListItem>
                    <ListItemAvatar>
                      <Avatar>
                        <RestoreIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={date
                        .toISOString()
                        .replace(/T/, ' ') // replace T with a space
                        .replace(/\..+/, '')}
                    />
                    <ListItemIcon>
                      <IconButton aria-label="download">
                        <GetAppIcon />
                      </IconButton>
                    </ListItemIcon>
                    <ListItemIcon>
                      <IconButton aria-label="delete">
                        <DeleteIcon />
                      </IconButton>
                    </ListItemIcon>
                  </ListItem>
                );
              })
            : ''}
          <Divider component="li" />
          <ListItem>
            <ListItemIcon>
              <StorageIcon />
            </ListItemIcon>
            <ListItemText
              id="switch-list-label-wifi"
              primary="Hot Storage"
              secondary="Toggle Hot Storage"
            />
            <ListItemSecondaryAction>
              <Switch
                edge="end"
                onChange={handleToggle('wifi')}
                checked={checked.indexOf('wifi') !== -1}
                inputProps={{ 'aria-labelledby': 'switch-list-label-wifi' }}
              />
            </ListItemSecondaryAction>
          </ListItem>

          <ListItem>
            <ListItemText
              id="switch-list-label-allowUnfreeze"
              primary="Allow Unfreeze"
              secondary="if data isn't available in the Hot Storage, it's allowed to be feeded by Cold Storage if available."
              inset
            />
            <ListItemSecondaryAction>
              <Switch
                edge="end"
                onChange={handleToggle('allowUnfreeze')}
                checked={checked.indexOf('allowUnfreeze') !== -1}
                inputProps={{ 'aria-labelledby': 'switch-list-label-wifi' }}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <StorageIcon />
            </ListItemIcon>
            <ListItemText
              id="switch-list-label-cold"
              primary="Cold Storage"
              secondary="Toggle Cold Storage"
            />
            <ListItemSecondaryAction>
              <Switch
                edge="end"
                onChange={handleToggle('cold')}
                checked={checked.indexOf('cold') !== -1}
                inputProps={{ 'aria-labelledby': 'switch-list-label-wifi' }}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <Paper>
              <Stepper
                alternativeLabel
                activeStep={
                  checked.indexOf('cold') !== -1
                    ? selectedValue && selectedValue.modDates
                      ? (new Date() -
                          selectedValue.modDates[
                            selectedValue.modDates.length - 1
                          ]) /
                        1000 /
                        Math.floor(Math.random() * (8 - 6) + 5)
                      : 5
                    : 0
                }
                connector={<QontoConnector />}
              >
                {steps.map((label) => (
                  <Step key={label}>
                    <StepLabel StepIconComponent={QontoStepIcon}>
                      {label}
                    </StepLabel>
                  </Step>
                ))}
              </Stepper>
            </Paper>
          </ListItem>
          <ListItem>
            <ListItemText
              id="switch-list-label-repFactor"
              primary="Replication Factor"
              secondary="repFactor indicates the desired amount of active deals with different miners to store the data."
              inset
            />
            <ListItemSecondaryAction>
              <Input
                className={classes.input}
                value={value}
                margin="dense"
                onChange={handleInputChange}
                onBlur={handleBlur}
                inputProps={{
                  step: 1,
                  min: 0,
                  max: 100,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <ListItemText
              id="switch-list-label-dealMinDuration"
              primary="Deal Min Duration"
              secondary="dealMinDuration indicates the minimum duration to be used when making new deals."
              inset
            />
            <ListItemSecondaryAction>
              <Input
                className={classes.input}
                value={dealMinDuration}
                margin="dense"
                onChange={handleInputChangeDealMinDuration}
                onBlur={handleBlur}
                inputProps={{
                  step: 10,
                  min: 0,
                  max: 10000,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <ListItemText
              id="switch-list-label-renew"
              primary="Renew"
              secondary="Enabled indicates that deal-renewal is enabled for this File."
              inset
            />
            <ListItemSecondaryAction>
              <Switch
                edge="end"
                onChange={handleToggle('renew')}
                checked={checked.indexOf('renew') !== -1}
                inputProps={{ 'aria-labelledby': 'switch-list-label-renew' }}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <ListItemText
              id="switch-list-label-maxPrice"
              primary="Max Price"
              secondary="Maximum price that will be spent per RepFactor to store the data in units of attoFIL per GiB per epoch."
              inset
            />
            <ListItemSecondaryAction>
              <Input
                className={classes.input}
                value={maxPrice}
                margin="dense"
                onChange={handleInputChangeMaxPrice}
                onBlur={handleBlur}
                inputProps={{
                  step: 100,
                  min: 0,
                  max: 1000000,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </ListItemSecondaryAction>
          </ListItem>
          <ListItem>
            <ListItemText
              id="switch-list-label-fastRetrieval"
              primary="Fast Retrieval"
              secondary="FastRetrieval indicates that created deals should enable the fast retrieval feature."
              inset
            />
            <ListItemSecondaryAction>
              <Switch
                edge="end"
                onChange={handleToggle('fastRetrieval')}
                checked={checked.indexOf('fastRetrieval') !== -1}
                inputProps={{
                  'aria-labelledby': 'switch-list-label-fastRetrieval',
                }}
              />
            </ListItemSecondaryAction>
          </ListItem>
        </List>
      </DialogContent>
      <DialogActions>
        <Button autoFocus onClick={handleClose} color="secondary">
          Cancel
        </Button>
        <Button onClick={handleClose} color="primary" autoFocus>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export const VFSBrowser: React.FC<VFSProps> = React.memo((props) => {
  const [open, setOpen] = React.useState(false);
  const [selectedValue, setSelectedValue] = React.useState<
    CustomFileData | undefined
  >(undefined);

  const handleClickOpen = (fileData: CustomFileData) => {
    setOpen(true);
    setSelectedValue(fileData);
  };

  const handleClose = (value: string) => {
    setOpen(false);
    // setSelectedValue(value);
  };

  const {
    fileMap,
    currentFolderId,
    setCurrentFolderId,
    resetFileMap,
    deleteFiles,
    moveFiles,
    createFolder,
    createFile,
    createSyncFolder,
    syncFile,
    refreshLocalFolder,
  } = useCustomFileMap();
  const files = useFiles(fileMap, currentFolderId);
  const folderChain = useFolderChain(fileMap, currentFolderId);
  const handleFileAction = useFileActionHandler(
    setCurrentFolderId,
    deleteFiles,
    moveFiles,
    createFolder,
    createFile,
    createSyncFolder,
    syncFile,
    refreshLocalFolder,
    handleClickOpen
  );
  const fileActions = useMemo(
    () => [
      ChonkyActions.CreateFolder,
      ChonkyActions.DeleteFiles,
      ChonkyActions.UploadFiles,
      AddSyncFolder,
      SynchronizeFolder,
      RefreshLocalFolder,
      Properties,
    ],
    []
  );

  // ipcRenderer.on('selected-folder', (_event: any, fileName: string) => {
  //   console.log(fileName);
  //   const folderName: string = path.basename(path.dirname(fileName));
  //   if (folderName) createFolder(folderName);
  // });
  
  // load the main file broswer.
  return (
    <div style={{ height: '90%', margin: '8px' }}>
      <div style={{ height: '100%' }}>
        <SimpleDialog
          selectedValue={selectedValue}
          open={open}
          onClose={handleClose}
        />
        
        <FileBrowser
          files={files}
          folderChain={folderChain}
          fileActions={fileActions}
          onFileAction={handleFileAction}
          defaultFileViewActionId={ChonkyActions.EnableListView.id}
        >
          <FileNavbar />
          <FileToolbar />
          <FileList />

          <FileContextMenu />
        </FileBrowser>

      </div>
    </div>
  );
});
