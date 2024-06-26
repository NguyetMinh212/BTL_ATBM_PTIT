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


window.addEventListener('load', function () {
    //Inject to html the bubble warning
    var bubble = document.createElement('div');
    bubble.id = 'malicious-url-warning';
    bubble.style.position = 'fixed';
    bubble.style.bottom = '10px';
    bubble.style.right = '10px';
    bubble.style.padding = '10px';
    bubble.style.backgroundColor = 'red';
    bubble.style.color = 'white';
    bubble.style.zIndex = '9999';
    bubble.style.display = 'none';
    bubble.innerHTML = 'Warning: This URL is malicious!';
    document.body.appendChild(bubble);
    console.log('Sending URL to background script...');
    let url = window.location.href;

    console.log(url);
    //post using fetch
    fetch("https://lich.quochung.cyou/predict", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
    }).then(response => response.json()).then(data => {
        console.log(data.data.prediction);
        if (data.data.prediction == 'bad') {
            //display warning message
            document.getElementById('malicious-url-warning').style.display = 'block';
        }
    }).catch((error) => {
        console.error('Error:', error);
    });
    console.log('URL sent to background script');
}, false);

//test: http://sharepoint-file-doc.s3-web.jp-tok.cloud-object-storage.appdomain.cloud/

