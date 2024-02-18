document.addEventListener("DOMContentLoaded", function() {
    setTimeout(function() {
        var gifContainer = document.getElementById('gif-container');
        gifContainer.style.display = 'block';

        setTimeout(function() {
            gifContainer.style.display = 'none';
            var signout_button=document.getElementById("sign_out");
            signout_button.style.display='none'
        }, 1000); // Stop after 1 second
        todarkmode()
    }, 2000); // Start after 2 seconds // 15000 milliseconds = 15 seconds
});
function todarkmode(){
    
    var navbar=document.getElementById("navbar")
    navbar.style.backgroundColor="#0c0d0f";
    var logo=document.getElementsByClassName("logo")
    logo[0].src="/frontend/images/logo.png"
    var navbar2 = document.getElementById("myNavbar");
    setTimeout(function(){
       
        if (navbar2) {
            navbar2.classList.remove("bg-light", "navbar-light");
            navbar2.classList.add("bg-dark", "navbar-dark");
        }
    },2000);
   
            navbar2.style.setProperty('background-color', '#0c0d0f', 'important');
            document.body.style.setProperty('background-color', '#0c0d0f', 'important');
    setTimeout(function(){
    var signout_button=document.getElementById("sign_out");
    signout_button.style.display='flex'
    signout_button.style.backgroundColor="white";
    signout_button.style.color="black"},3000);

}