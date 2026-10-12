import 'virtual:uno.css';
import './app.css';
import { render } from 'solid-js/web';
import App from '@/App';
import { takeEntry } from '@/stores/account';

// /login, /account and one-time #email= / #ssh= tokens: rewritten before anything reads the URL
takeEntry();

render(() => <App />, document.getElementById('app')!);
