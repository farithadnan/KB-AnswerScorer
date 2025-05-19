###### Get the latest update
Write-Host "Getting the latest update!" -ForegroundColor White -BackgroundColor DarkGreen

###### Shut Down and Remove The Related Containers
Write-Host "Shutting Down and Removing The Related Containers....." -ForegroundColor White -BackgroundColor DarkRed -NoNewline

# Command
docker-compose down

Write-Host "The Related Containers Has Shut Down and Removed Successfully!" -ForegroundColor White -BackgroundColor DarkGreen

###### Rebuild the Docker Image
Write-Host "Rebuilding the Docker Image....." -ForegroundColor White -BackgroundColor DarkRed

# Command
docker-compose build

Write-Host "Docker Image Rebuild Successfully!" -ForegroundColor White -BackgroundColor DarkGreen


###### Create and Run the Containers Based on the New Image
Write-Host "Creating and Running the Containers....." -ForegroundColor White -BackgroundColor DarkRed

# Command
docker-compose up -d

Write-Host "The Containers Has Been Created and Run Successfully!" -ForegroundColor White -BackgroundColor DarkGreen
