package example.jllarraz.com.minutiae.utils

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import androidx.preference.PreferenceManager
import androidx.activity.result.ActivityResultLauncher
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import example.jllarraz.com.minutiae.permission.Permission
import example.jllarraz.com.minutiae.permission.PermissionStatus

object PermissionUtils {


    fun checkPermissionStatus(activity: Activity, permission:String, isFirstTime:Boolean): PermissionStatus {
        val hasPermission = ContextCompat.checkSelfPermission(activity,
            permission) == PackageManager.PERMISSION_GRANTED
        if(hasPermission){
            return PermissionStatus.GRANTED
        } else {
            if(ActivityCompat.shouldShowRequestPermissionRationale(activity, permission)){
                return PermissionStatus.DENIED
            }else{
                //It returns false the first time that you ask for a permission and if the user clicked in never ask again
                if(isFirstTime) {
                    return PermissionStatus.DENIED
                }else{
                    return PermissionStatus.REVOKED
                }
            }
        }
    }

    fun checkPermissions(activity: Activity, permissions:ArrayList<String>):ArrayList<Permission>{
        val defaultSharedPreferences = PreferenceManager.getDefaultSharedPreferences(activity)
        val permissionsStatus:ArrayList<Permission> = ArrayList()

        for(permission in permissions){
            val isPermissionsHasBeenRequested = defaultSharedPreferences.getBoolean(permission, false)
            permissionsStatus.add(
                Permission(
                    permission,
                    checkPermissionStatus(activity, permission, !isPermissionsHasBeenRequested)
                )
            )
        }
        return permissionsStatus
    }

    fun allPermissionsGranted(permissions:ArrayList<Permission>):Boolean{
        for (permission in permissions){
            if(permission.status!= PermissionStatus.GRANTED){
                return false
            }
        }
        return true
    }

    fun arePermissionsRevoked(permissions:ArrayList<Permission>):Boolean{
        for (permission in permissions){
            if(permission.status == PermissionStatus.REVOKED){
                return true
            }
        }
        return false
    }

    fun getPermissions(permissions:ArrayList<Permission>, status: PermissionStatus):ArrayList<String>{
        val list:ArrayList<String> = ArrayList()
        for (permission in permissions){
            if(permission.status == status){
                list.add(permission.name)
            }
        }
        return list
    }

    fun requestPermissionsToUser(context: Context, permissions:ArrayList<Permission>, activityResultLauncher: ActivityResultLauncher<Array<String>>){
        val permissionsDenied = getPermissions(permissions, PermissionStatus.DENIED)
        requestPermissionsDeniedToUser(context, permissionsDenied, activityResultLauncher)
    }

    fun requestPermissionsDeniedToUser(context: Context, permissionsDenied:ArrayList<String>, activityResultLauncher: ActivityResultLauncher<Array<String>>){
        setPermissionsAlreadyRequested(context = context, permissions = permissionsDenied, alreadyRequested = true)
        activityResultLauncher.launch(permissionsDenied.toTypedArray())
    }

    fun setPermissionAlreadyRequested(context: Context, permission:String, alreadyRequested:Boolean){
        val defaultSharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        val edit = defaultSharedPreferences.edit()
        edit.putBoolean(permission, alreadyRequested)
        edit.commit()
    }

    fun setPermissionsAlreadyRequested(context: Context, permissions:ArrayList<String>, alreadyRequested:Boolean){
        val defaultSharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        val edit = defaultSharedPreferences.edit()
        for(permission in permissions){
            edit.putBoolean(permission, alreadyRequested)
        }
        edit.commit()
    }
}
