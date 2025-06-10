function fetchObjects() {
    fetch('/detected_objects')
        .then(response => response.json())
        .then(data => {
            const objectList = document.getElementById("object-list");
            objectList.innerHTML = "";

            if (data.objects.length === 0) {
                const li = document.createElement("li");
                li.textContent = "No objects detected yet.";
                objectList.appendChild(li);
            } else {
                data.objects.forEach(obj => {
                    const li = document.createElement("li");
                    li.textContent = obj.charAt(0).toUpperCase() + obj.slice(1);
                    objectList.appendChild(li);
                });
            }
        })
        .catch(err => console.error("Error fetching detected objects:", err));
}

// Refresh every 2 seconds
setInterval(fetchObjects, 2000);

