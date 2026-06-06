/* Bootstraps cytoscape + fcose plugin and exposes cytoscape on window
   so app.js (non-module) can use it. Split from index.html inline script
   to allow strict CSP (script-src 'self' without 'unsafe-inline'). */
import cytoscape from '/static/vendor/cytoscape.esm.js';
import fcose from '/static/vendor/cytoscape-fcose.esm.js';
cytoscape.use(fcose);
window.cytoscape = cytoscape;
