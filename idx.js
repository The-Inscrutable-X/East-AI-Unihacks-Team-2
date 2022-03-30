const form = document.querySelector("#form");
const out = document.querySelector("#query-results");
const q = document.querySelector("#query");
const tgt = document.querySelector("#tgt-sentences");
const serverStatus = document.querySelector("#status");

form.addEventListener('submit', e => {
    e.preventDefault();
    serverStatus.innerHTML = 'Status: Awaiting response from server...';
    console.log('http://localhost:5000?query=' + q.value + "&tgt-sentences=" + parseInt(tgt.value));
    if (parseInt(tgt.value) > 10) {
        alert("too many sentences requested");
        return;
    };
    fetch('http://localhost:5000?query=' + q.value + "&tgt-sentences=" + tgt.value, {
        method: 'POST',
    })
    .then(res => {
        return res.json()
    })
    .then(obj => {
        console.log(obj);
        serverStatus.innerHTML = 'Status: Fetch Successful! ';
        for (i in obj) {
            out.innerHTML += "<p>" + obj[i].toString() + "</p>";
        }
    })
    .catch(error => {
        serverStatus.innerHTML = 'Status: Error ';
        console.log(error);
    })

    return false;
});