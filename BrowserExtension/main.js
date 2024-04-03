// ==UserScript==
// @name         Malicious URL Dectector
// @version      0.0
// @match        *://*/*
// @grant        none
// ==/UserScript==

// This script is injected into every page that the user visits
// It will send the URL of the page to the background script
// The background script will then check the URL against a list of malicious URLs
// If the URL is malicious, the background script will send a message to the content script
// The content script will then display a warning message to the user


window.addEventListener('load', function() {
    console.log('Sending URL to background script...');
    var url = window.location.href;
    //post using fetch
    fetch("https://lich.quochung.cyou/predict", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
    }).then(response => response.json()).then(data => {
        console.log(data);
    }).catch((error) => {
        console.error('Error:', error);
    });
    console.log('URL sent to background script');
}, false);