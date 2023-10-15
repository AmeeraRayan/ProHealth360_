// script.js

document.addEventListener("DOMContentLoaded", function () {
    const scrollButton = document.getElementById("scrollButton");
    const gifContainer = document.getElementById("gifContainer");

    // Show the GIF container on button hover
    scrollButton.addEventListener("mouseenter", function () {
        gifContainer.style.display = "block";
    });

    // Hide the GIF container when the mouse leaves the button
    scrollButton.addEventListener("mouseleave", function () {
        gifContainer.style.display = "none";
    });

    // Add a click event listener to the button for scrolling
    scrollButton.addEventListener("click", function () {
        const targetSection = document.getElementById("targetSection");

        // Scroll to the target section
        targetSection.scrollIntoView({ behavior: "smooth" });
    });
});
