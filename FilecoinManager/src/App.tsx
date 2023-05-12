import React from 'react';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import './App.global.css';

import { createStyles, makeStyles, Theme } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import CloseIcon from '@material-ui/icons/Close';
import MinimizeIcon from '@material-ui/icons/Minimize';

import { setChonkyDefaults, FullFileBrowser } from 'chonky';
import { ChonkyIconFA } from 'chonky-icon-fontawesome';
import icon from '../assets/icon.svg';

import { VFSBrowser } from './demo1';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      flexGrow: 1,
    },
    menuButton: {
      marginRight: theme.spacing(2),
      '-webkit-app-region': 'no-drag',
    },
    title: {
      flexGrow: 1,
    },
    titleBar: {
      // backgroundColor: '#09f',
      '-webkit-app-region': 'drag',
    },
  })
);

const Hello = () => {
  const classes = useStyles();
  setChonkyDefaults({ iconComponent: ChonkyIconFA });
  return (
    // <div style={{ height: '100vh' }}>
    //   <VFSBrowser instanceId="test" />
    // </div>
    <div className={classes.root} style={{ height: '100vh' }}>
      <AppBar
        position="static"
        style={{ 'margin-bottom': '8px' }}
        className={classes.titleBar}
      >
        <Toolbar variant="dense">
          <IconButton
            edge="start"
            className={classes.menuButton}
            color="inherit"
            aria-label="menu"
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" className={classes.title}>
            Filecoin Manager
          </Typography>
          <IconButton
            edge="start"
            className={classes.menuButton}
            color="inherit"
            aria-label="minimize"
          >
            <MinimizeIcon />
          </IconButton>
          <IconButton
            edge="start"
            className={classes.menuButton}
            color="inherit"
            aria-label="close"
          >
            <CloseIcon />
          </IconButton>
        </Toolbar>
      </AppBar>
      <VFSBrowser instanceId="test" />
    </div>
  );
};

export default function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" component={Hello} />
      </Switch>
    </Router>
  );
}
